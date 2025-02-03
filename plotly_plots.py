import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import logging
import os
import webbrowser
from pathlib import Path
import plotly.express as px

from preprocessing import * 
from mpl_plots import *
from constants import *

def normalize_spectrum(arr):
    arr = np.asarray(arr)
    max_val = np.max(arr)
    assert max_val != 0
    return 100 * arr / max_val

def plot_acoustic_spectra(dataset, start, end):
    fig = go.Figure()
    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    
    # Main spectra plot
    for i, sensor_id in enumerate(all_sensors):
        filtered_dataset = dataset.sel(sensor=sensor_id).where(
            dataset.sel(sensor=sensor_id)['base'].notnull() &
            (dataset['datetime'] >= start) &
            (dataset['datetime'] <= end),
            drop=True
        )
        if filtered_dataset['datetime'].shape[0] == 0:
            continue
        averaged_spectrum = normalize_spectrum(
            get_non_nan_spectrum(filtered_dataset.mean(dim='datetime')['spectrum'].values)
        )
        spectrum_len = averaged_spectrum.shape[0]
        freq_factor = filtered_dataset['frequency_scaling_factor'].values[0]
        freq_start = filtered_dataset['frequency_start_index'].values[0]
        frequencies = [(bin+freq_start)*freq_factor for bin in range(spectrum_len)]
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=averaged_spectrum,
            name=f'Sensor {sensor_id}',
            marker_color=colors[i],
            opacity=0.7,
            line_shape='spline',
            hovertemplate='(%{y:.1f}%, %{x:.1f} Hz<extra></extra>)',
        ))
    
    fig.update_layout(
        title='Beehive acoustic spectra',
        xaxis=dict(
            title='Frequency, Hz',
            range=[0, 850],
            tickfont=dict(size=10),
            tickangle=0,
        ),
        yaxis=dict(
            title='Relative Amplitude, %',
            range=[0, 100],
            tickfont=dict(size=10),
            automargin=False,
            tickangle=0,
        ),
        hovermode='closest',
        dragmode='zoom',
        legend=dict(
            yanchor="top",
            xanchor="right",
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        uirevision=True,
        selectionrevision=True,
        margin=dict(l=25, r=25, t=50, b=25)
    )
    
    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ACOUSTIC_SPECTRA_HTML, 'w') as file:
        file.write(rendered_html)
    logging.info(f"HTML file {ACOUSTIC_SPECTRA_HTML} was created!")
    return rendered_html

def plot_time_slider(dataset, start, end):
    fig = go.Figure()
    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    
    for i, sensor_id in enumerate(all_sensors):
        filtered_dataset = dataset.sel(sensor=sensor_id).where(
            dataset.sel(sensor=sensor_id)['base'].notnull() &
            (dataset['datetime'] >= start) &
            (dataset['datetime'] <= end),
            drop=True
        )
        if filtered_dataset['datetime'].shape[0] == 0:
            continue
        times = filtered_dataset['datetime'].values
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[0] * len(times),
                mode='markers',
                marker=dict(
                    color=colors[i], 
                    size=4
                ),
                hovertemplate='%{x}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title='Time range selection',
        xaxis=dict(
            type='date',
            rangeslider=dict(
                visible=True,
                thickness=0.2,
            ),
            tickfont=dict(size=10),
            tickangle=0,
            automargin=False, # Fixes width changes when new tick label appears while sliding
        ),
        yaxis=dict(visible=False),
        hovermode='closest',
        dragmode='zoom',
        showlegend=False,
        height=200, 
        margin=dict(l=25, r=25, t=50, b=25)
    )
    
    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TIME_SLIDER_HTML, 'w') as file:
        file.write(rendered_html)
    logging.info(f"HTML file {TIME_SLIDER_HTML} was created!")
    return rendered_html

def plot_temperature_humidity(dataset, start, end):
    fig = go.Figure()
    start = start.astimezone(HELSINKI_TZ)
    end = end.astimezone(HELSINKI_TZ)

    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    last_datetime = max(dataset["datetime"].values).astimezone(HELSINKI_TZ)
    
    for i, sensor_id in enumerate(all_sensors):
        filtered_dataset = dataset.sel(sensor=sensor_id).where(
                    dataset.sel(sensor=sensor_id)['base'].notnull() &
                    (dataset['datetime'] >= start) &
                    (dataset['datetime'] <= end),
                    drop=True
        )
        if filtered_dataset['datetime'].shape[0] == 0:
            continue
        temperatures = filtered_dataset['temperature'].values
        humidities = filtered_dataset['humidity'].values
        times = filtered_dataset['datetime'].values
        floating_hours_time_ago = [(last_datetime - t).total_seconds()/3600 for t in times]
        max_ago = max(floating_hours_time_ago)
        opacities = [0.2 + 0.8 * (1 - t/max_ago) for t in floating_hours_time_ago]
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
            name=f'Sensor {sensor_id}',
                hovertemplate=(
                    '%{fullData.name}<br>'
                    'Temperature: %{x:.1f}°C<br>'
                    'Humidity: %{y:.1f}%<br>'
                    'DateTime: %{customdata[0]}<br>'
                    'Ago from most recent datapoint: %{customdata[1]:.1f} h<extra></extra>'
                ),
                customdata=list(zip(
                    ['{}'.format(t.astimezone(HELSINKI_TZ)) for t in times],
                    floating_hours_time_ago,
                ))
            )
        )
    
    fig.update_layout(
        title='Beehive temperature and humidity',
        xaxis_title='Temperature, °C',
        yaxis_title='Relative Humidity, %',
        hovermode='closest',
        legend=dict(
            yanchor="top",     
            xanchor="right",
            bgcolor='rgba(255, 255, 255, 0.5)',
        ),
        uirevision=True,
        selectionrevision=True,
        margin=dict(l=25, r=25, t=50, b=25)
    )

    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TEMPERATURE_HUMIDIY_HTML, 'w') as file:
        file.write(rendered_html)
    logging.info(f"HTML file {TEMPERATURE_HUMIDIY_HTML} was created!")
    return rendered_html

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    sensors = [20, 21, 46, 109]
    csv_files = download_csv_if_needed(
        sensors,
        HELSINKI_4DAYS_AGO.astimezone(UTC_TZ),
        HELSINKI_NOW.astimezone(UTC_TZ),
        DATA_DIR
    ) 
    dataset = load_dataset(csv_files)
    
    # Create three individual plots:
    acoustic_spectra_html = plot_acoustic_spectra(dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW)
    time_slider_html = plot_time_slider(dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW)
    temperature_humidity_html = plot_temperature_humidity(dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW)
    
    # Combine the three plots into a single HTML output
    with open(PLOTLY_COMBINED_HTML, 'w') as file:
        file.write(acoustic_spectra_html + time_slider_html + temperature_humidity_html)
        logging.info(f"HTML file {PLOTLY_COMBINED_HTML} created")

    webbrowser.open(Path(PLOTLY_COMBINED_HTML).absolute().as_uri())
