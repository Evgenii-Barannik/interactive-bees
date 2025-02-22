from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import os
import logging
import matplotlib as mpl
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from lmfit import Model, Parameters

from constants import *
from preprocessing import get_info_total, load_dataset, show_image, download_csv_if_needed
from plotly_plots import normalize_spectrum

def calculate_pearson_distance(spectra):
    distances = pdist(spectra, metric='correlation')
    distance_matrix = squareform(distances)
    return distance_matrix

def calculate_fisher_information_distance(spectra):
    num_spectra = spectra.shape[0]
    distance_matrix = np.zeros((num_spectra, num_spectra))
    for i in range(num_spectra):
        for j in range(i, num_spectra):
            spectrum_i = spectra[i] / np.sum(spectra[i])
            spectrum_j = spectra[j] / np.sum(spectra[j])
            fisher_distance = np.sqrt(np.sum((np.sqrt(spectrum_i) - np.sqrt(spectrum_j)) ** 2))
            distance_matrix[i, j] = fisher_distance
            distance_matrix[j, i] = fisher_distance 
    return distance_matrix

def calculate_angular_distance(spectra):
    inner_products = spectra @ spectra.T
    norms = np.linalg.norm(spectra, axis=1)
    norm_matrix = np.outer(norms, norms)
    cosine_similarity = (inner_products / norm_matrix)
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    angular_distance = np.arccos(cosine_similarity)/np.pi
    return angular_distance

def calculate_euclidean_distance(spectra):
    distances = pdist(spectra, metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix

def format_time_to_helsinki(x, _):
    dt = datetime.fromtimestamp(x, tz=HELSINKI_TZ) # Timestamps are converted to Helsinki timezone
    return dt.strftime('%d %H:%M')

def get_ticks_for_helsinki_tz(start, end, step_in_hours):
    start = start.astimezone(HELSINKI_TZ)
    end = end.astimezone(HELSINKI_TZ)
    assert start < end
    midnight_before_start = start.replace(hour=0, minute=0, second=0, microsecond=0) 
    midnight_after_end = end.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    ticks = [copy.deepcopy(midnight_before_start)]
    moving = midnight_before_start
    while (moving < midnight_after_end):
        moving = moving + pd.Timedelta(hours=step_in_hours)
        ticks.append(moving)
    return ticks

def get_extended_datetimes(ds, sensor_id, start, end):
    ds = ds.where(
        (ds.sensor == sensor_id),
        drop = True,
        other = 0
    )
    datetimes_before_including = ds.where(
        (ds['datetime'] <= start),
        drop = True,
        other = 0
    )['datetime'].values

    # If not points before found, use start time as extension:
    if len(datetimes_before_including) != 0:
        logging.info(f"Previous point used as pre-extension.")
        pre_extension = max(datetimes_before_including)
    else:
        logging.info(f"Start point used as pre-extension.")
        pre_extension = start
        
    datetimes_after_including = ds.where(
        (ds['datetime'] >= end), 
        drop = True,
        other = 0
    )['datetime'].values

    # If not points after found, use end time as extension:
    if len(datetimes_after_including) != 0:
        logging.info(f"Next point used as post-extension.")
        post_extension = min(datetimes_after_including)
    else:
        logging.info(f"End used as post-extension.")
        post_extension = end

    pre_extension = pre_extension.astimezone(HELSINKI_TZ)
    post_extension = post_extension.astimezone(HELSINKI_TZ)
    logging.info(f"Datetime for pre-extension:   {pre_extension}")
    logging.info(f"Datetime for post-extension:  {post_extension}")
    # If there are points at start or end, we would have already included it, thus we use strict checks:
    datetimes_in_range = ds.where(
        (ds['datetime'] > start) & 
        (ds['datetime'] < end), 
        drop = True,
        other = 0
    )['datetime'].values

    extended_datetimes = np.append(pre_extension, datetimes_in_range)
    extended_datetimes = np.append(extended_datetimes, post_extension)
    return extended_datetimes

def plot_similarity(ds, start, end, output_path, name_overide=None):
    logging.info(f"Plotting similarity for requested range\nSTART:   {start}\nEND:     {end}")
    assert start < end # No graph is possible otherwise
    images = []
    all_sensors = np.unique(ds.sensor)

    for sensor_id in all_sensors:
        filtered_by_timerange = ds.where (
            (ds.sensor == sensor_id) &
            (ds['datetime'] > start) &  # Strict filtering since start and end may be used as extenstions
            (ds['datetime'] < end),
            drop = True,
            other = 0
        )
        if len(filtered_by_timerange['datetime'].values) == 0: # There must be some datapoints to plot a graph
            continue
        measurement_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in filtered_by_timerange['datetime'].values])
        num_of_datapoints = len(measurement_datetimes)
        logging.info(f"\nFor sensor {sensor_id}:")
        logging.info(f"First datapoints: {min(measurement_datetimes)}")
        logging.info(f"Last datapoint:   {max(measurement_datetimes)}")
        logging.info(f"Num of datapoints: {num_of_datapoints}")

        # Spectra for extension datapoints do not matter. If there are spectra for those points, those spectra will not influence plot.
        # But extension datetimes will be used to construct leftmost and rightmost Voronoi edges.
        extended_datetimes = get_extended_datetimes(ds, sensor_id, start, end)
        extended_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in extended_datetimes])
        assert extended_datetimes[0] <= start # pre-extension datetime should be before start or at start
        assert extended_datetimes[-1] >= end # post-extension datetime should be after end or at end

        unix_epochs= np.array([t.timestamp() for t in extended_datetimes]) # Unix/Posix epochs (counted from UTC)
        voronoi_edges = (unix_epochs[:-1] + unix_epochs[1:]) / 2
        x_edges = voronoi_edges
        y_edges = voronoi_edges
        leftmost_edge = pd.to_datetime(voronoi_edges[0], unit='s', utc=True).tz_convert('Europe/Helsinki')
        rightmost_edge = pd.to_datetime(voronoi_edges[-1], unit='s', utc=True).tz_convert('Europe/Helsinki')
        logging.info(f"Voronoi edges from:  {leftmost_edge}\nVoronoi edges to:    {rightmost_edge}")

        spectra = np.vstack(filtered_by_timerange['spectrum'].values)

        # Compute distances, will be used to color Voronoi cells
        pearson_matrix = calculate_pearson_distance(spectra)
        fisher_matrix = calculate_fisher_information_distance(spectra)
        angular_matrix = calculate_angular_distance(spectra)
        euclidean_matrix = calculate_euclidean_distance(spectra)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colormap = mpl.colormaps["coolwarm"] 
        im1 = axes[0, 0].pcolormesh(x_edges, y_edges, pearson_matrix, cmap=colormap, shading="auto")
        axes[0, 0].set_title('Pearson distance')
        plt.colorbar(im1, ax=axes[0, 0]) 

        im2 = axes[0, 1].pcolormesh(x_edges, y_edges, euclidean_matrix, cmap=colormap, shading="auto")
        axes[0, 1].set_title('Euclidean distance')
        plt.colorbar(im2, ax=axes[0, 1]) 

        im3 = axes[1, 0].pcolormesh(x_edges, y_edges, angular_matrix, cmap=colormap, shading="auto")
        axes[1, 0].set_title('Angular distance')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].pcolormesh(x_edges, y_edges, fisher_matrix, cmap=colormap, shading="auto")
        axes[1, 1].set_title('Fisher distance')
        plt.colorbar(im4, ax=axes[1, 1])

        # Data is ploted using unix epochs
        # Ticks are set using unix epochs
        # Tick labels show datetimes in Helsinki timezone
        datetimes_for_ticks = get_ticks_for_helsinki_tz(start, end, 12) 
        epochs_for_ticks = [x.timestamp() for x in datetimes_for_ticks] # Unix epochs

        # Settings for Axes
        formatter = FuncFormatter(format_time_to_helsinki)
        for ax in axes.flat:
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()
            ax.set_xticks(epochs_for_ticks, labels=datetimes_for_ticks)
            ax.set_yticks(epochs_for_ticks, labels=datetimes_for_ticks)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=30, ha='right')
            for epoch in np.array([t.timestamp() for t in measurement_datetimes]): # Dots for timestamps
                ax.scatter(epoch, epoch, color='black', s=1)

        annotation_to_place_on_plot = "Acoustic similarity measures for signal from sensor {}\n{}".format(
                sensor_id,
                get_info_total(filtered_by_timerange)
        )
        fig.text(0.6, 0.88, annotation_to_place_on_plot, ha='right', fontsize=14)
        plt.tight_layout(pad=2.5)
        fig.subplots_adjust(top=0.88)

        if name_overide:
            img_pathname = os.path.join(output_path, name_overide)
        else:
            img_pathname = os.path.join(output_path, f"similarity-measures-sensor-{sensor_id}.png")

        images.append(img_pathname) 
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300)
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    return images

def plot_similarity_example():
    sensors = [21]
    start = datetime(2024, 8, 11, 0, tzinfo = HELSINKI_TZ)
    end = datetime(2024, 8, 14, 0, 0, tzinfo = HELSINKI_TZ)
    csv_files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
            )
    dataset = load_dataset(csv_files)
    similarity_plots = plot_similarity(
            dataset,
            start,
            end,
            OUTPUT_DIR,
            "similarity_example.png"
    )
    return similarity_plots

FITTING_MODEL = [
        {
            'type': 'background',
            'slope_guess': 0,
            'intercept_guess': 10,
        },
        {
            'type': 'peak',
            'center_range': (70, 155),
            'amplitude_guess': 70,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (175, 275),
            'amplitude_guess': 40,
            'fwhm_guess': 60
        },
        {  
            'type': 'peak',
            'center_range': (300, 350),
            'amplitude_guess': 40,
            'fwhm_guess': 50
        },
        {  
            'type': 'peak',
            'center_range': (370, 440),
            'amplitude_guess': 20,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (450, 500),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (500, 520),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (550, 650),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        }
    ]

def linear_background(x, slope, intercept):
    return slope * x + intercept

def gaussian(x, amplitude, center, fwhm):
    sigma = fwhm / 2.355
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def create_model():
    model = Model(linear_background, prefix='bg_')
    for i in range(1, len(FITTING_MODEL)):
        model += Model(gaussian, prefix=f'g{i}_')
    return model

def plot_gaussians(ds, start, end, output_path, name_overide=None):
    images = []
    all_sensors = np.unique(ds.sensor)

    for sensor_id in all_sensors:
        filtered_ds = ds.where (
            (ds.sensor == sensor_id) &
            (ds['datetime'] > start) & 
            (ds['datetime'] < end),
            drop = True,
            other = 0
        )
        if len(filtered_ds['datetime'].values) == 0: # There must be some datapoints to plot a graph
            continue
        raw_spectra = np.vstack(filtered_ds['spectrum'].values)
        averaged_spectrum = normalize_spectrum(np.nanmean(raw_spectra, axis=0))
        spectrum_len = len(averaged_spectrum)

        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start  = filtered_ds['frequency_start_index'].values[0]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectrum_len)])
                                               
        x_full = frequencies
        y_full = averaged_spectrum
        
        mask_min = 60
        mask_max = 650
        mask = (x_full >= mask_min) & (x_full <= mask_max)
        x_masked = x_full[mask]
        y_masked = y_full[mask]
        
        model = create_model()
        params = Parameters()
        bg = FITTING_MODEL[0]
        params.add('bg_slope', value=bg['slope_guess'], vary=False)
        params.add('bg_intercept', value=bg['intercept_guess'], min=0, max=np.min(y_masked))
        for i, cfg in enumerate(FITTING_MODEL):
            if cfg['type'] == 'peak':
                prefix = f'g{i}_'
                params.add(f'{prefix}amplitude', value=cfg['amplitude_guess'], min=0)
                params.add(f'{prefix}center', value=np.mean(cfg['center_range']), min=cfg['center_range'][0], max=cfg['center_range'][1])
                params.add(f'{prefix}fwhm', value=cfg['fwhm_guess'], min=10, max=100)
        result = model.fit(y_masked, params, x=x_masked)
        
        components = result.eval_components(x=x_masked)
        
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        ax1.set_ylabel('Amplitude', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
        
        ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        ax2.set_ylabel('Residuals', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12, length=6, width=1.5)
        
        ax1.plot(x_full, y_full, 'b-', label='Acoustic spectrum')
        ax1.plot(x_masked, result.best_fit, 'k--', label='Total model')
        ax1.plot(x_masked, components['bg_'], 'c--', label='Linear background')
        for i, (name, comp) in enumerate(components.items(), 1):
            if name.startswith('g'):
                ax1.plot(x_masked, comp, label=f'Gauss peak {i-1}')
        ax1.set_ylim(0, 100)
        ax1.set_xlim(0, 850)
        ax1.legend(fontsize=10)
        
        residuals = y_masked - result.best_fit
        ax2.plot(x_masked, residuals, 'r-', label='Residuals', alpha=0.7)
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)

        p = result.params
        annotation_to_place_on_plot = [
                "Fit for averaged acoustic data from sensor {}\n{}".format(
                sensor_id,
                get_info_total(filtered_ds))
        ]
        annotation_to_place_on_plot.extend(["Background: {:.2f}x + {:.2f}".format(p['bg_slope'].value, p['bg_intercept'].value)])
        annotation_to_place_on_plot.extend([f"Gauss peak N: Height, Center, FWHM"])
        for i in range(1, len(FITTING_MODEL)):
            annotation_to_place_on_plot.extend([
                f"Gauss peak {i}: {p[f'g{i}_amplitude'].value:.1f}, {p[f'g{i}_center'].value:.1f} Hz, {p[f'g{i}_fwhm'].value:.1f} Hz"
            ])
        ax1.annotate('\n'.join(annotation_to_place_on_plot), xy=(0.52, 0.55), xycoords='axes fraction', fontsize=10)
        plt.tight_layout()
        
        if name_overide:
            img_pathname = os.path.join(output_path, name_overide)
        else:
            img_pathname = os.path.join(output_path, f"gaussians-sensor-{sensor_id}.png")

        images.append(img_pathname) 
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    return images

def plot_gauss_example():
    sensors = [116]

    start = datetime(2025, 2, 13, 0, tzinfo = HELSINKI_TZ)
    end = datetime(2025, 2, 17, 0, 0, tzinfo = HELSINKI_TZ)
    csv_files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
            )
    filtered_ds = load_dataset(csv_files, True, start, end)
    gauss_plots = plot_gaussians(filtered_ds, OUTPUT_DIR, "gauss_example.png")
    return gauss_plots

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    similarity_example = plot_similarity_example()
    show_image(similarity_example[0])
    gauss_example = plot_gauss_example()
    show_image(gauss_example[0])

