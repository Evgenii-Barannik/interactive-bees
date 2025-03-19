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

def plot_evolution(ds, start, end, output_path, name_overide=None):
    logging.info(f"Plotting peak evolution for requested range\nSTART:   {start}\nEND:     {end}")
    assert start < end
    images = []
    all_sensors = np.unique(ds.sensor)

    for sensor_id in all_sensors:
        filtered_by_timerange = ds.where(
            (ds.sensor == sensor_id) &
            (ds['datetime'] > start) & 
            (ds['datetime'] < end),
            drop=True,
            other=0
        )
        if len(filtered_by_timerange['datetime'].values) == 0:
            continue

        measurement_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in filtered_by_timerange['datetime'].values])
        num_of_datapoints = len(measurement_datetimes)
        logging.info(f"\nFor sensor {sensor_id}:")
        logging.info(f"First datapoints: {min(measurement_datetimes)}")
        logging.info(f"Last datapoint:   {max(measurement_datetimes)}")
        logging.info(f"Num of datapoints: {num_of_datapoints}")
        
        extended_datetimes = get_extended_datetimes(ds, sensor_id, start, end)
        extended_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in extended_datetimes])
        assert extended_datetimes[0] <= start # pre-extension datetime should be before start or at start
        assert extended_datetimes[-1] >= end # post-extension datetime should be after end or at end

        unix_epochs= np.array([t.timestamp() for t in extended_datetimes]) # Unix/Posix epochs (counted from UTC)
        voronoi_edges = (unix_epochs[:-1] + unix_epochs[1:]) / 2
        leftmost_edge = pd.to_datetime(voronoi_edges[0], unit='s', utc=True).tz_convert('Europe/Helsinki')
        rightmost_edge = pd.to_datetime(voronoi_edges[-1], unit='s', utc=True).tz_convert('Europe/Helsinki')
        logging.info(f"Voronoi edges from:  {leftmost_edge}\nVoronoi edges to:    {rightmost_edge}")

        spectra = np.vstack(filtered_by_timerange['spectrum'].values)
        freq_factor = filtered_by_timerange['frequency_scaling_factor'].values[0]
        freq_start = filtered_by_timerange['frequency_start_index'].values[0]
        spectra_len = spectra.shape[1]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectra_len)])

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.grid(True, alpha=0.3)
        gauss_count = sum(1 for cfg in FITTING_MODEL if cfg['type'] == 'peak')

        for j, spectrum in enumerate(spectra):
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
                    ax.add_patch(rect)                   
                    ax.scatter(center, measurement_datetime.timestamp(), color=color, edgecolors='black')
                    plot_rectangles(ax, gauss_count, fill=False)

        ax.set_xlabel('Frequency, Hz', fontsize=14)
        ax.set_ylabel('Time', fontsize=14)
        ax.set_xlim(0, 700)
        
        ax.yaxis.set_major_formatter(FuncFormatter(format_time_to_helsinki))

        handles = []
        for i, cfg in enumerate(FITTING_MODEL):
            if cfg['type'] == 'peak':
                color = COLORMAP_FOR_GAUSSIANS(i / gauss_count)
                handles.append(mpatches.Patch(color=color, label=f'Gauss peak {i}'))
        # ax.legend(handles=handles, title='Evolution of FWHM for gauss peaks')
        fig.legend(title='Evolution of FWHM for gauss peaks', bbox_to_anchor=(0.98, 0.98), handles=handles, fontsize=9)
        plt.tight_layout()
        
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
    sensors = [116]
    start = datetime(2025, 2, 13, 0, 0, tzinfo=HELSINKI_TZ)
    end = datetime(2025, 2, 13, 6, 0, tzinfo=HELSINKI_TZ)
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
