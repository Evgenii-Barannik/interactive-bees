import numpy as np
from lmfit import Model, Parameters
import logging
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import time

from datetime import datetime
from constants import *
from plotly_plots import normalize_spectrum
from preprocessing import get_info_total, load_dataset, show_image, download_csv_if_needed
from similarity_plot import get_ticks_for_helsinki_tz, format_time_to_helsinki, get_extended_datetimes
from gauss_plot import plot_rectangles, fit_model

def plot_evolution(ds, start, end, output_path, name_overide=None):
    start_time = time.time()
    logging.info(f"Plotting peak evolution for requested range\nSTART:   {start}\nEND:     {end}")
    assert start < end
    images = []
    all_sensors = np.unique(ds.sensor)

    for sensor_id in all_sensors:
        filtered_ds = ds.where(
            (ds.sensor == sensor_id) &
            (ds['datetime'] > start) & 
            (ds['datetime'] < end),
            drop=True,
            other=0
        )
        if len(filtered_ds['datetime'].values) == 0:
            continue

        measurement_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in filtered_ds['datetime'].values])
        extended_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in get_extended_datetimes(ds, sensor_id, start, end)])
        assert extended_datetimes[0] <= start # pre-extension datetime should be before start or at start
        assert extended_datetimes[-1] >= end # post-extension datetime should be after end or at end

        unix_timestamps = np.array([t.timestamp() for t in extended_datetimes]) # Unix/Posix epochs (counted from UTC)
        voronoi_edges = (unix_timestamps[:-1] + unix_timestamps[1:]) / 2
        oldest_edge = pd.to_datetime(voronoi_edges[0], unit='s', utc=True).tz_convert('Europe/Helsinki')
        latest_edge = pd.to_datetime(voronoi_edges[-1], unit='s', utc=True).tz_convert('Europe/Helsinki')
        logging.info(f"Voronoi edges from:  {oldest_edge}\nVoronoi edges to:    {latest_edge}")

        spectra = np.vstack(filtered_ds['spectrum'].values)
        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start = filtered_ds['frequency_start_index'].values[0]
        spectra_len = spectra.shape[1]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectra_len)])

        fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 8),
                gridspec_kw={'width_ratios': [8, 1]},
                sharey=True
        )
        ax0 = axes[0]
        ax1 = axes[1]
        ax0.grid(True, alpha=0.3)
        gauss_count = sum(1 for cfg in FITTING_MODEL if cfg['type'] == 'peak')

        rmse_values = []
        for j, spectrum in enumerate(spectra):
            normalized_spectrum = normalize_spectrum(spectrum, frequencies, NORMALIZATION_LIMIT)
            (_, result, _, rmse) = fit_model(frequencies, normalized_spectrum)
            rmse_values.append(rmse) 
            measurement_datetime = measurement_datetimes[j]
            measurment_timestamp = measurement_datetime.timestamp()
            start_edge = voronoi_edges[j]
            end_edge = voronoi_edges[j+1]
            
            freq_grid = np.linspace(0, 700, 175) # Increase num of points to have better resolution
            model_intensity = result.eval(x=freq_grid)  # Evaluate model at frequency grid
            
            cmap = plt.cm.turbo
            cmap_colors = cmap(np.linspace(0, 1, cmap.N))
            cmap_colors[:, -1] = 0.8  # Set alpha channel to 0.8
            cmap = mcolors.ListedColormap(cmap_colors)
            norm = mcolors.Normalize(vmin=0, vmax=np.max(model_intensity))
            
            # Draw sections of horizontal colored stripe 
            for i in range(len(freq_grid)-1):
                model_intensity_at_freq = model_intensity[i]
                color = cmap(norm(model_intensity_at_freq))
                
                ax0.fill_betweenx(
                    [start_edge, end_edge],   # y-values (time range)
                    freq_grid[i],             # x-values (frequency)
                    freq_grid[i+1],           
                    color=color,
                    alpha=0.8,
                    linewidth=0
                )
            
            # Draw letters
            for i, cfg in enumerate(FITTING_MODEL):
                if cfg['type'] == 'peak':
                    prefix = f'g{i}_'
                    if f'{prefix}center' in result.params:
                        center = result.params[f'{prefix}center'].value
                        letter = chr(64 + i)
                        ax0.text(center, measurment_timestamp, letter, fontsize=7, 
                                 ha='center', va='center', weight='bold', color='black'
                                 )
            
        # Draw ranges 
        plot_rectangles(ax0, gauss_count, fill=False)

        ax1.scatter(rmse_values, [d.timestamp() for d in measurement_datetimes], color='grey', edgecolors='black')
        ax1.plot(rmse_values, [d.timestamp() for d in measurement_datetimes], 'k-')

        datetimes_for_ticks = get_ticks_for_helsinki_tz(start, end)
        timestamps_for_ticks = [d.timestamp() for d in datetimes_for_ticks]
        ax0.set_yticks(timestamps_for_ticks, labels=datetimes_for_ticks)
        ax0.yaxis.set_major_formatter(FuncFormatter(format_time_to_helsinki))

        ax0.set_xlabel('Frequency, Hz', fontsize=14)
        ax0.set_ylabel('DateTime', fontsize=14)
        ax0.set_xlim(0, 700)
        ax0.set_ylim(start.timestamp(), end.timestamp())
        ax1.set_xlabel('RMSE', fontsize=14)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax0, label='Intensity, %')
        cbar.ax.tick_params(labelsize=10)

        handles = []
        for i, cfg in enumerate(FITTING_MODEL):
            if cfg['type'] == 'peak':
                letter = chr(64 + i)
                center = cfg['center_range']
                patch = mpatches.Patch(color='None', label=f'{letter}: center of peak in {center} Hz range')
                handles.append(patch)
        
        datapoints_info = "\nEvolution of total model intensity (color)\nand gauss peak center positions (letters)\nfor fitted individual acoustic spectra\n{}Sensor: {}".format(get_info_total(filtered_ds), sensor_id, FITTING_WINDOW_MIN, FITTING_WINDOW_MAX)
        patch = mpatches.Patch(color='None', label=datapoints_info) 
        handles.append(patch)
        window_text = f"\nWindow used for fitting: {FITTING_WINDOW_MIN} to {FITTING_WINDOW_MAX} Hz"
        patch = mpatches.Patch(color='None', label=window_text)
        handles.append(patch)
        fig.legend(bbox_to_anchor=(0.98, 0.98), handles=handles, fontsize=9)

        patch = mpatches.Patch(color='None', label=f"Gauss peak N: Center, FWHM, Amplitude")
        handles.append(patch) 
        plt.tight_layout()
        plt.subplots_adjust(right=0.74)

        if name_overide:
            img_pathname = os.path.join(output_path, name_overide)
        else:
            img_pathname = os.path.join(output_path, f"evolution-sensor-{sensor_id}.png")
        
        images.append(img_pathname)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    
    total_time = time.time() - start_time
    logging.info(f"Total plot creation time: {total_time:.2f} seconds")
    return images

def plot_peak_evolution_example():
    sensors = [21]
    start = datetime(2024, 8, 11, 0, tzinfo = HELSINKI_TZ)
    end = datetime(2024, 8, 14, 0, 0, tzinfo = HELSINKI_TZ)
    csv_files = download_csv_if_needed(
        sensors,
        start.astimezone(UTC_TZ),
        end.astimezone(UTC_TZ),
        DATA_DIR
    )
    filtered_ds = load_dataset(csv_files, True, start, end)
    evolution_plots = plot_evolution(filtered_ds, start, end, OUTPUT_DIR, "peak_evolution_example.png")
    return evolution_plots

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    evolution_example = plot_peak_evolution_example()
    show_image(evolution_example[0])
