from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from zoneinfo import ZoneInfo
import copy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

from preprocessing import *
import matplotlib.colors as mcolors

def get_portland_colormap(num = 256):
    plotly_colors = px.colors.sample_colorscale("Portland", num)
    colors = []
    for rgb_str in plotly_colors:
        rgb_values = rgb_str.strip('rgb()').split(',')
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
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    dt = datetime.fromtimestamp(x, tz=helsinki_tz) # Timestamps are converted to Helsinki timezone
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

def get_text_for_legend(filtered_dataset):
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    num_of_datapoints = filtered_dataset['datetime'].shape[-1]
    start_time = str(filtered_dataset['datetime'].values[0].astimezone(helsinki_tz))
    end_time = str(filtered_dataset['datetime'].values[-1].astimezone(helsinki_tz))
    text_for_legend = f"First datapoint: {start_time}\nLast datapoint: {end_time}\nNumber of datapoints: {num_of_datapoints}\n" 
    return text_for_legend

def plot_continuous_correlations(ds, start, end, output_path):
    assert start < end
    utc_tz = ZoneInfo('UTC')
    text_for_html_annotation = ""
    distance_images = []
    for sensor in ds["sensor"].values:
        filtered_dataset = ds.sel(sensor=sensor).where(
                ds.sel(sensor=sensor)['base'].notnull() &
                (ds['datetime'] >= start) &
                (ds['datetime'] <= end),
                drop=True
        )

        measurment_datetimes = filtered_dataset['datetime'].values
        end = pd.to_datetime(end.replace(microsecond=0).astimezone(utc_tz).isoformat()) # Shadowing
        all_datetimes = np.append(measurment_datetimes, end)
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

        # # Compute distances, will be used to color Voronoi cells
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
        datetimes_for_ticks = get_ticks_between(start, end) # Datetimes in UTC or Helsinki time (depends on input)
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
            for epoch in np.array([t.timestamp() for t in measurment_datetimes]): # Dots for timestamps
                ax.scatter(epoch, epoch, color='black', s=0.5)

        text_for_html_annotation = text_for_html_annotation + "For sensor {}:<br>{}<br>".format(sensor, get_text_for_legend(filtered_dataset)).replace('\n', '<br>')
        fig.text(0.35, 0.97, f"Signal from sensor {sensor}", fontsize=18)
        plt.tight_layout(pad=3)  

        # Save plot
        img_pathname = os.path.join(output_path, f"distance-measures-sensor-{sensor}.png")
        distance_images.append(img_pathname) 
        plt.savefig(img_pathname, dpi=300)
        plt.close()
        logging.info(f"Plot {img_pathname} was created!")
    return distance_images, text_for_html_annotation

def plot_correlations(correlations, frequencies, output_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=frequencies, y=correlations.real,
                  name="Real part",
                  line=dict(color='blue')),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=frequencies, y=correlations.imag,
                  name="Imaginary part",
                  line=dict(color='red')),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Frequency (Hz)")

    # Set y-axes titles
    fig.update_yaxes(title_text="Real part", secondary_y=False)
    fig.update_yaxes(title_text="Imaginary part", secondary_y=True)

    fig.update_layout(
        title="Correlation Function",
        hovermode='x unified'
    )

    # Save to HTML for interactive viewing
    html_path = os.path.join(output_path, "correlations.html")
    png_path = os.path.join(output_path, "correlations.png")
    
    fig.write_html(html_path)
    fig.write_image(png_path)
    
    return html_path, png_path

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    # Second sensor with different spectrun size is included for testing stripping arrays with NaNs
    sensors = [21, 109]
    data_dir = "data"
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz) 
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=4)

    files = download_csv_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(files)
    correlations, _ = plot_continuous_correlations(dataset, helsinki_days_ago, helsinki_now, "plots")
    show_image(correlations[-1])
