import numpy as np
from lmfit import Model, Parameters
import logging
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

from datetime import datetime
from constants import *
from plotly_plots import normalize_spectrum
from preprocessing import get_info_total, load_dataset, show_image, download_csv_if_needed
from similarity_plot import get_ticks_for_helsinki_tz, format_time_to_helsinki, get_extended_datetimes
from gauss_plot import plot_rectangles, fit_model

def calculate_acoustic_power(spectrum, frequencies):
    delta_f = np.abs(frequencies[1] - frequencies[0])
    spectral_power_density = np.abs(spectrum)**2
    total_power = np.sum(spectral_power_density) * delta_f
    return total_power

def plot_evolution(ds, start, end, output_path, name_overide=None):
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
        num_of_datapoints = len(measurement_datetimes)
        logging.info(f"\nFor sensor {sensor_id}:")
        logging.info(f"First datapoints: {min(measurement_datetimes)}")
        logging.info(f"Last datapoint:   {max(measurement_datetimes)}")
        logging.info(f"Num of datapoints: {num_of_datapoints}")
        
        extended_datetimes = get_extended_datetimes(ds, sensor_id, start, end)
        extended_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in extended_datetimes])
        assert extended_datetimes[0] <= start # pre-extension datetime should be before start or at start
        assert extended_datetimes[-1] >= end # post-extension datetime should be after end or at end

        unix_epochs = np.array([t.timestamp() for t in extended_datetimes]) # Unix/Posix epochs (counted from UTC)
        voronoi_edges = (unix_epochs[:-1] + unix_epochs[1:]) / 2
        leftmost_edge = pd.to_datetime(voronoi_edges[0], unit='s', utc=True).tz_convert('Europe/Helsinki')
        rightmost_edge = pd.to_datetime(voronoi_edges[-1], unit='s', utc=True).tz_convert('Europe/Helsinki')
        logging.info(f"Voronoi edges from:  {leftmost_edge}\nVoronoi edges to:    {rightmost_edge}")

        spectra = np.vstack(filtered_ds['spectrum'].values)
        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start = filtered_ds['frequency_start_index'].values[0]
        spectra_len = spectra.shape[1]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectra_len)])

        fig, axes = plt.subplots(
                1, 2,
                figsize=(14, 8),
                gridspec_kw={'width_ratios': [6, 1]},
                sharey=True
        )
        ax0 = axes[0]
        ax1 = axes[1]
        ax0.grid(True, alpha=0.3)
        gauss_count = sum(1 for cfg in FITTING_MODEL if cfg['type'] == 'peak')

        acoustic_power_values = []
        for j, spectrum in enumerate(spectra):
            acoustic_power = calculate_acoustic_power(spectrum, frequencies)
            acoustic_power_values.append(acoustic_power)
            normalized_spectrum = normalize_spectrum(spectrum)
            (_, result, _, _) = fit_model(frequencies, normalized_spectrum)
            p = result.params
            
            # Plot each peak
            for i, cfg in enumerate(FITTING_MODEL):
                if cfg['type'] == 'peak':
                    prefix = f'g{i}_'
                    color = COLORMAP_FOR_GAUSSIANS(i / gauss_count)
                    center = p[f'{prefix}center'].value
                    fwhm = p[f'{prefix}fwhm'].value
                    
                    measurement_datetime = measurement_datetimes[j]
                    left_edge = voronoi_edges[j]
                    right_edge = voronoi_edges[j+1]
                    duration = right_edge - left_edge
 
                    rect = patches.Rectangle(
                        (center - fwhm/2, left_edge), 
                        fwhm, 
                        duration, 
                        linewidth=1, 
                        edgecolor=color, 
                        facecolor=color, 
                        alpha=0.4
                    )
                    ax0.add_patch(rect)                   
                    ax0.scatter(center, measurement_datetime.timestamp(), color=color, edgecolors='black')
                    plot_rectangles(ax0, gauss_count, fill=False)


        if sensor_id >= 100:
            ax1.scatter(acoustic_power_values, [d.timestamp() for d in measurement_datetimes], color='grey', edgecolors='black')
            ax1.plot(acoustic_power_values, [d.timestamp() for d in measurement_datetimes], 'k-')

        datetimes_for_ticks = get_ticks_for_helsinki_tz(start, end)
        timestamps_for_ticks = [d.timestamp() for d in datetimes_for_ticks]
        ax0.set_yticks(timestamps_for_ticks, labels=datetimes_for_ticks)
        ax0.yaxis.set_major_formatter(FuncFormatter(format_time_to_helsinki))

        ax0.set_xlabel('Frequency, Hz', fontsize=14)
        ax0.set_ylabel('Time', fontsize=14)
        ax0.set_xlim(0, 700)
        ax0.set_ylim(start.timestamp(), end.timestamp())
        ax1.set_xlabel('Acoustic power, AU', fontsize=14)

        handles = []
        for i, cfg in enumerate(FITTING_MODEL):
            if cfg['type'] == 'peak':
                color = COLORMAP_FOR_GAUSSIANS(i / gauss_count)
                center = cfg['center_range']
                handles.append(mpatches.Patch(color=color, label=f'Gauss peak {i}: {center} Hz'))
        patch = mpatches.Patch(color='None', label=f"Gauss peak N: range for center position")
        handles.append(patch) 
        datapoints_info = "\nEvolution of FWHM (colored areas) and\ncenter positions (dots) of gauss peaks\nfor fitted individual acoustic spectra\n{}Sensor: {}".format(get_info_total(filtered_ds), sensor_id, FITTING_WINDOW_MIN, FITTING_WINDOW_MAX)
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
