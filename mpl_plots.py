from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import plotly.express as px
import copy
import os
from datetime import datetime
from matplotlib.ticker import FuncFormatter

from preprocessing import *
from constants import *

def get_portland_colormap(num = 256):
    plotly_colors = px.colors.sample_colorscale("Portland", num)
    colors = []
    for rgb_str in plotly_colors:
        rgb_values = str(rgb_str).strip('rgb()').split(',')
        r, g, b = [int(x) for x in rgb_values]
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(hex_color)
    if num == 256:
        return mcolors.LinearSegmentedColormap.from_list('portland', colors)
    else:
        return colors

def strip_nan_columns(arr):
    arr = np.asarray(arr)
    return arr[:, ~np.all(np.isnan(arr), axis=0)]

def calculate_pearson_distance(spectra):
    distances = pdist(spectra, metric='correlation')
    distance_matrix = squareform(distances)
    return distance_matrix

def calculate_cosine_distance(spectra):
    distances = pdist(spectra, metric='cosine')
    distance_matrix = squareform(distances)
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
    return dt.strftime('%H:%M %d')

def get_ticks_between(start, end):
    assert start < end
    midnight_before_start = start.replace(hour=0, minute=0, second=0, microsecond=0) 
    midnight_after_end = end.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    ticks = [copy.deepcopy(midnight_before_start)]
    moving = midnight_before_start
    while (moving < midnight_after_end):
        moving = moving + pd.Timedelta(hours=12)
        ticks.append(moving)
    return ticks

def plot_similarity(ds, start, end, output_path, name_overide = None):
    assert start < end
    images = []
    for sensor in ds["sensor"].values:
        filtered_dataset = ds.sel(sensor=sensor).where(
                ds.sel(sensor=sensor)['base'].notnull() &
                (ds['datetime'] >= start) &
                (ds['datetime'] <= end),
                drop=True
        )

        measurement_datetimes = filtered_dataset['datetime'].values
        end = pd.to_datetime(end.replace(microsecond=0).astimezone(UTC_TZ).isoformat()) # Shadowing
        all_datetimes = np.append(measurement_datetimes, end)
        unix_epochs = np.array([t.timestamp() for t in all_datetimes]) # Unix/Posix epochs (counted from UTC)
        
        # Find edges for 1D Voronoi tesselation
        voronoi_edges = (unix_epochs[:-1] + unix_epochs[1:]) / 2
        leftmost_edge  = unix_epochs[0] - (unix_epochs[1] - unix_epochs[0]) / 2
        voronoi_edges_extended = np.concatenate([
            [leftmost_edge],
            voronoi_edges,
        ])
        x_edges = voronoi_edges_extended
        y_edges = voronoi_edges_extended

        # Compute distances, will be used to color Voronoi cells
        spectra = filtered_dataset['spectrum'].values
        spectra = strip_nan_columns(spectra)

        pearson_matrix = calculate_pearson_distance(spectra)
        cosine_matrix = calculate_cosine_distance(spectra)
        angular_matrix = calculate_angular_distance(spectra)
        euclidean_matrix = calculate_euclidean_distance(spectra)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        colormap = get_portland_colormap()
        im1 = axes[0, 0].pcolormesh(x_edges, y_edges, pearson_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
        axes[0, 0].set_title('Pearson distance')
        plt.colorbar(im1, ax=axes[0, 0]) 

        im2 = axes[0, 1].pcolormesh(x_edges, y_edges, euclidean_matrix, cmap=colormap, shading="auto")
        axes[0, 1].set_title('Euclidean distance')
        plt.colorbar(im2, ax=axes[0, 1]) 

        im3 = axes[1, 0].pcolormesh(x_edges, y_edges, angular_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
        axes[1, 0].set_title('Angular distance')
        plt.colorbar(im3, ax=axes[1, 0])

        im4 = axes[1, 1].pcolormesh(x_edges, y_edges, cosine_matrix, cmap=colormap, shading="auto", vmin=0, vmax=1)
        axes[1, 1].set_title('Cosine distance')
        plt.colorbar(im4, ax=axes[1, 1])

        # Data is ploted using unix epochs
        # Ticks are set using unix epochs
        # Tick labels show datetimes in Helsinki timezone
        datetimes_for_ticks = get_ticks_between(start, end) 
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
                ax.scatter(epoch, epoch, color='black', s=0.5)

        annotation_to_place_on_plot = "Similarity measures for signal from sensor {}:\n{}".format(
                sensor,
                get_info_about_filtered_datapoints(filtered_dataset)
        )
        fig.text(0.6, 0.88, annotation_to_place_on_plot, ha='right', fontsize=14)
        plt.tight_layout(pad=2.5)
        fig.subplots_adjust(top=0.88)

        if name_overide:
            img_pathname = os.path.join(output_path, name_overide)
        else:
            img_pathname = os.path.join(output_path, f"similarity-measures-sensor-{sensor}.png")

        images.append(img_pathname) 
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300)
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    return images

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    helsinki_start = datetime(2024, 8, 11, tzinfo = HELSINKI_TZ)
    helsinki_end = datetime(2024, 8, 15, tzinfo = HELSINKI_TZ)

    sensors = [21]
    csv_files = download_csv_if_needed(sensors, helsinki_start, helsinki_end, DATA_DIR) 
    dataset = load_dataset(csv_files)
    correlations = plot_similarity(dataset, helsinki_start, helsinki_end, OUTPUT_DIR, "similarity_example.png")
    show_image(correlations[0])
