import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import plotly.express as px

from preprocessing import * 
from mpl_plots import *

def filter(ds, sensor_id, start, end):
    filtered = ds.sel(sensor=sensor_id).where(
            ds.sel(sensor=sensor_id)['base'].notnull() &
            (ds['datetime'] >= start) &
            (ds['datetime'] <= end),
            drop=True
    )
    return filtered

def normalize_spectrum(arr):
    arr = np.asarray(arr)
    max_val = np.max(arr)
    assert max_val != 0
    return 100 * arr / max_val

def create_averaged_plot_html(dataset, start, end):
    fig = go.Figure()
    text_for_html_annotation = """
    Each shown spectrum is created by averaging
    spectra from multiple datapoints and then normalizing resulting spectrum.<br><br>
    """
    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    traces = []
    for i, sensor in enumerate(all_sensors):
        filtered_dataset = filter(dataset, sensor, start, end)
        num_of_datapoints = filtered_dataset['datetime'].shape[0]
        if num_of_datapoints == 0:
            logging.info(f"Dataset for sensor {sensor} is empty. Skipping plotting")
            continue
        averaged_spectrum = normalize_spectrum(get_non_nan_spectrum(filtered_dataset.mean(dim='datetime')['spectrum'].values))
        text_for_html_annotation = text_for_html_annotation + "For sensor {}:<br>{}<br>".format(sensor, get_text_for_legend(filtered_dataset)).replace('\n', '<br>')
        spectrum_len_non_nan = averaged_spectrum.shape[0]
        frequency_scaling_factor = filtered_dataset['frequency_scaling_factor'].values[0]
        frequency_start_index = filtered_dataset['frequency_start_index'].values[0]
        frequencies = [
            (bin+frequency_start_index)*frequency_scaling_factor
            for bin in range(spectrum_len_non_nan) 
        ]
        traces.append(
            go.Scatter(
                x=frequencies,
                y=averaged_spectrum,
                name=f'Sensor {sensor}',
                marker_color=colors[i],
                opacity=0.7,
                hovertemplate='(%{y:.1f}%, %{x:.1f} Hz<extra></extra>)', #<extra></extra> removes sensor number(it is obvious anyway)
                 
                line_width=3,
                hoveron='points+fills'
            )
        )

    fig.add_traces(traces)
    fig.update_layout(
        title=dict(
            text='Beehive acoustic spectra',
        ),
        xaxis_title='Frequency, Hz',
        yaxis_title='Relative Amplitude, %',
        xaxis_range=[0, 850],
        yaxis_range=[0, 100],
        hovermode='closest',
        legend=dict(
            yanchor="top",     
            xanchor="right",
            bgcolor='rgba(255, 255, 255, 0.5)',  # White background with 0.7 opacity
        ),
        margin=dict(l=25, r=25, t=75, b=0, pad=0),
    )
    
    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    return rendered_html, text_for_html_annotation

def create_temp_humidity_plot_html(dataset, sensors, start, end):
    fig = go.Figure()
    text_for_html_annotation = f"""
    Temperature-Humidity phase plot.<br>
    Points with more vivid colors are more recent.<br><br>
    """
    colors = px.colors.sample_colorscale("Portland", len(sensors))
    helsinki_tz = ZoneInfo('Europe/Helsinki')
    last_datetime = max(dataset["datetime"].values).astimezone(helsinki_tz)

    for i, sensor in enumerate(sensors):
        filtered_dataset = filter(dataset, sensor, start, end)
        
        # Get temperature and humidity data
        temperatures = filtered_dataset['temperature'].values
        humidities = filtered_dataset['humidity'].values
        times = filtered_dataset['datetime'].values
        
        # Calculate time ago for hover text and opacity
        floating_hours_time_ago = [(last_datetime - t).total_seconds()/3600 for t in times]
        max_ago = max(floating_hours_time_ago)
        opacities = [0.2 + 0.8 * (1 - t/max_ago) for t in floating_hours_time_ago]
        text_for_html_annotation = text_for_html_annotation + "For sensor {}:<br>{}<br>".format(sensor, get_text_for_legend(filtered_dataset)).replace('\n', '<br>')
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=humidities,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[i],
                    opacity=opacities
                ),
                name=f'Sensor {sensor}',
                hovertemplate=(
                    '%{fullData.name}<br>'
                    'Temperature: %{x:.1f}°C<br>'
                    'Humidity: %{y:.1f}%<br>'
                    'DateTime: %{customdata[0]}<br>'
                    'Ago from most recent datapoint: %{customdata[1]:.1f} h<extra></extra>'
                ),
                customdata=list(zip(
                    ['{}'.format(t.astimezone(helsinki_tz)) for t in times],
                    floating_hours_time_ago,
                ))
            )
        )

    fig.update_layout(
        title=dict(
            text='Beehive temperature and humidity',
        ),
        xaxis_title='Temperature, °C',
        yaxis_title='Relative Humidity, %',
        hovermode='closest',
        legend=dict(
            yanchor="top",     
            xanchor="right",
            bgcolor='rgba(255, 255, 255, 0.5)',
        ),
        margin=dict(l=25, r=25, t=75, b=0, pad=0)
    )

    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    return rendered_html, text_for_html_annotation

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=4)
    helsinki_24h_ago = helsinki_now - pd.Timedelta(days=1)

    data_dir = "data"
    plots_dir = "plots"
    html_path = "plots/plotly_plots.html"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    sensors = [20, 21, 46, 109]
    csv_files = download_csv_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(csv_files)
    with open(html_path, 'w') as file:
        averaged_plot_html, _ = create_averaged_plot_html(dataset, helsinki_days_ago, helsinki_now)
        phase_plot_html, _ = create_temp_humidity_plot_html(dataset, dataset["sensor"].values, helsinki_24h_ago, helsinki_now)
        file.write(averaged_plot_html + phase_plot_html)
    webbrowser.open(Path(html_path).absolute().as_uri())
