import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import logging
import os
import webbrowser
import plotly.express as px
import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots

from constants import *
from preprocessing import load_dataset, download_csv_if_needed 

COMMON_MARGIN = dict(l=25, r=25, t=50, b=50)
CONFIG = {
    'displaylogo': False,
    'responsive': True,
    'modeBarButtonsToRemove': [ 'lasso2d' ]
}

# We use annotations instead of plot titles to control gap between title and plot
ANNOTATION_DEFAULTS = dict(
    x=0,
    y=1.02,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(size=16)
)

LEGEND_CONFIG = dict(
    tracegroupgap=0,
    font=dict(size=8),
    title_text='Sensor', 
)

def normalize_spectrum(arr):
    arr = np.asarray(arr)
    max_val = np.max(arr)
    assert max_val != 0
    return 100 * arr / max_val

def plot_parallel_selector(ds, return_fig=False):
    sensors = ds.sensor.values
    unique_sensors = np.unique(sensors)
    sensor_to_index_map = {sensor: i for (i, sensor) in enumerate(unique_sensors)}
    sensor_indices = [sensor_to_index_map[s] for s in sensors] 

    measurment_datetimes = [t.astimezone(HELSINKI_TZ) for t in ds.datetime.values]
    tick_datetimes = pd.date_range(
        start=min(measurment_datetimes),
        end=max(measurment_datetimes),
        periods=10,
        tz=HELSINKI_TZ
    )

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=sensor_indices,
                colorscale='Portland',
            ),
            dimensions = list([
                dict(
                    label = 'Sensor',
                    values = sensor_indices, 
                    tickvals = list(sensor_to_index_map.values()),
                    ticktext = list(sensor_to_index_map.keys()),
                    range = [0, len(unique_sensors)-1]   
                ),
                dict(
                    # Datetimes can not be used for values for this type of plot, so we use timestamps:
                    # https://github.com/plotly/plotly.py/issues/968
                    label = 'DateTime',
                    values = [t.timestamp() for t in ds.datetime.values],
                    tickvals = [t.timestamp() for t in tick_datetimes],
                    ticktext = [t.strftime('%d %b %H:%M') for t in tick_datetimes]
                ),
                dict(
                    label = 'Temperature,°C',
                    values = ds.temperature.values
                    ),
                dict(
                    label = 'Humidity, %',
                    values = ds.humidity.values
                    ),
            ])
        )
    )
    fig.update_layout(
        margin={**COMMON_MARGIN, "r":50},
    )

    if return_fig:
        return fig
    else:
        rendered_html = fig.to_html(config=CONFIG, include_plotlyjs=True, full_html=False)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(PARALLEL_SELECTOR_HTML, 'w') as file:
            file.write(rendered_html)
        logging.info(f"HTML file {PARALLEL_SELECTOR_HTML} was created!")
        return rendered_html

def plot_acoustic_spectra(ds, start, end, return_fig=False):
    fig = go.Figure()
    unique_sensors = np.unique(ds["sensor"].values)
    colors = px.colors.sample_colorscale("Portland", len(unique_sensors))
    
    for i, sensor_id in enumerate(unique_sensors):
        filtered_ds = ds.where(
            (ds.sensor == sensor_id) &
            (ds['datetime'] >= start) &
            (ds['datetime'] <= end),
            drop=True,
            other=0
        )
        if len(filtered_ds['datetime'].values) == 0:
            logging.info(f"No data for sensor {sensor_id}, skipping it.")
            continue

        raw_spectra = np.vstack(filtered_ds['spectrum'].values)
        raw_times = [
            t.astimezone(HELSINKI_TZ)
            for t in filtered_ds['datetime'].values
        ]

        averaged_spectrum = normalize_spectrum(np.nanmean(raw_spectra, axis=0))
        spectrum_len = len(averaged_spectrum)
        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start  = filtered_ds['frequency_start_index'].values[0]
        frequencies = [(bin+freq_start)*freq_factor for bin in range(spectrum_len)]

        fig.add_trace(go.Scatter(
            x=frequencies,
            y=averaged_spectrum,
            name=str(sensor_id),
            marker_color=colors[i],
            opacity=0.7,
            line_shape='spline',
            hovertemplate='(%{y:.1f}%, %{x:.1f} Hz<extra></extra>)',
            meta={
                "raw_spectra": raw_spectra.tolist(),
                "raw_times": raw_times
            }
        ))
    
    fig.update_layout(
        xaxis=dict(
            title='Frequency, Hz',
            range=[0, 850],
            tickangle=0,
            gridwidth=2
        ),
        yaxis=dict(
            title='Relative Amplitude, %',
            range=[0, 100],
            tickangle=0,
            gridwidth=2
        ),
        hovermode='closest',
        dragmode='zoom',
        legend=LEGEND_CONFIG,
        uirevision=True,
        selectionrevision=True,
        margin=COMMON_MARGIN,
    )
    fig.add_annotation(text="Acoustic spectra", **ANNOTATION_DEFAULTS)
    
    if return_fig:
        return fig
    else:
        rendered_html = fig.to_html(config=CONFIG, include_plotlyjs=True, full_html=False)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(ACOUSTIC_SPECTRA_HTML, 'w') as file:
            file.write(rendered_html)
        logging.info(f"HTML file {ACOUSTIC_SPECTRA_HTML} was created!")
        return rendered_html

def plot_time_slider(ds, return_fig=False):
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    
    all_sensors = np.unique(ds["sensor"].values)
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))

    for i, sensor_id in enumerate(all_sensors):
        filtered_ds = ds.where(
            ( ds.sensor == sensor_id ),
            drop=True,
            other=0
        )
        if len(filtered_ds['datetime']) == 0:
            continue

        times = [t.astimezone(HELSINKI_TZ) for t in filtered_ds['datetime'].values]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=filtered_ds.temperature.values,
                mode='lines+markers',
                marker=dict(
                    symbol='diamond-tall',
                    color=colors[i],
                    size=6,
                    opacity=0.5
                ),
                line=dict(
                    shape='spline',
                    color=colors[i],
                    width=1,
                ),
                name=f'{sensor_id}',
                legendgroup=f'sensor_{sensor_id}',
                showlegend=True,
                hovertemplate=(
                    '%{customdata}<br>'
                    'Temperature: %{y:.1f}°C<extra></extra>'
                ),
                customdata=[
                    t.astimezone(HELSINKI_TZ).strftime('%Y-%m-%d %H:%M:%S%z') 
                    for t in filtered_ds['datetime'].values
                ]
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=filtered_ds.humidity.values,
                mode='lines+markers',
                marker=dict(
                    symbol='circle',
                    color=colors[i],
                    size=6,
                    opacity=0.5
                ),
                line=dict(
                    shape='spline',
                    color=colors[i],
                    width=1,
                ),
                name=f'Sensor {sensor_id}',
                legendgroup=f'sensor_{sensor_id}',
                showlegend=False,
                hovertemplate=(
                    '%{customdata}<br>'
                    'Humidity: %{y:.1f}%<extra></extra>'
                ),
                customdata=[
                    t.astimezone(HELSINKI_TZ).strftime('%Y-%m-%d %H:%M:%S%z') 
                    for t in filtered_ds['datetime'].values
                ]
            ),
            row=2, col=1
        )

    fig.update_layout(
        hovermode='closest',
        dragmode='zoom',
        margin={**COMMON_MARGIN, "l":60},
        legend=LEGEND_CONFIG,
    )

    fig.update_yaxes(
        title_text="Temp,°C",
        title_font=dict(size=12),
        row=1, col=1
    )
     
    fig.update_yaxes(
        title_text="Humidity, %",
        title_font=dict(size=12),
        row=2, col=1
    )
    
    fig.update_xaxes(
        type='date',
        rangeslider=dict(
            visible=True,
            thickness=0.1,
        ),
        tickfont=dict(size=9),
        tickangle=0,
        row=2, col=1
    )
    
    fig.add_annotation(
        text="Time range selection", **ANNOTATION_DEFAULTS)

    if return_fig:
        return fig
    else:
        rendered_html = fig.to_html(config=CONFIG, include_plotlyjs=True, full_html=False)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(TIME_SLIDER_HTML, 'w') as file:
            file.write(rendered_html)
        logging.info(f"HTML file {TIME_SLIDER_HTML} was created!")
        return rendered_html

def plot_temperature_humidity(ds, return_fig=False):
    fig = go.Figure()
    all_sensors = np.unique(ds["sensor"].values)
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    last_datetime = max(ds["datetime"].values).astimezone(HELSINKI_TZ)
    
    for i, sensor_id in enumerate(all_sensors):
        filtered_ds = ds.where(
                (ds.sensor == sensor_id),
                drop=True,
                other=0,
        )
        if len(filtered_ds['datetime']) == 0:
            continue
        temperatures = filtered_ds['temperature'].values
        humidities = filtered_ds['humidity'].values
        times = filtered_ds['datetime'].values
        floating_hours_time_ago = [(last_datetime - t).total_seconds()/3600 for t in times]
        max_ago = max(floating_hours_time_ago)
        opacities = [0.2 + 0.8 * (1 - t/max_ago) for t in floating_hours_time_ago]
        fig.add_trace(
            go.Scatter(
                x=temperatures,
                y=humidities,
            # mode="lines+markers",
            mode="markers",
            marker=dict(
                symbol="arrow",
                size=15,
                angleref="previous",
                opacity=opacities,
                color=colors[i],
            ),
            name=str(sensor_id),
                # line=dict(
                #     shape='spline',
                #     smoothing=0.7,
                # ),
                hovertemplate=(
                    'Sensor %{fullData.name}<br>'
                    'Temperature: %{x:.1f}°C<br>'
                    'Humidity: %{y:.1f}%<br>'
                    'DateTime: %{customdata[0]|%d %b %H:%M}<br>'
                    'Ago from most recent: %{customdata[1]:.1f} h<extra></extra>'
                ),
                customdata=list(zip(
                    ['{}'.format(t.astimezone(HELSINKI_TZ)) for t in times],
                    floating_hours_time_ago,
                ))
            )
        )
    
    fig.update_layout(
        xaxis_title='Temperature, °C',
        yaxis_title='Relative Humidity, %',
        xaxis=dict(gridwidth=2),
        yaxis=dict(gridwidth=2),
        hovermode='closest',
        legend=LEGEND_CONFIG,
        uirevision=True,
        selectionrevision=True,
        margin=COMMON_MARGIN
    )
    fig.add_annotation(text="Temperature and humidity", **ANNOTATION_DEFAULTS)

    if return_fig:
        return fig
    else:
        rendered_html = fig.to_html(config=CONFIG, include_plotlyjs=True, full_html=False)
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

    sensors = [20, 21, 46, 116, 123]
    start = HELSINKI_4DAYS_AGO
    end = HELSINKI_NOW

    csv_files = download_csv_if_needed(
        sensors,
        start.astimezone(UTC_TZ),
        end.astimezone(UTC_TZ),
        DATA_DIR
    ) 
    dataset = load_dataset(csv_files)
    filtered_dataset = load_dataset(csv_files, True, start, end)
    
    time_slider_html = plot_time_slider(filtered_dataset)
    acoustic_spectra_html = plot_acoustic_spectra(filtered_dataset, start, end)
    temperature_humidity_html = plot_temperature_humidity(filtered_dataset)
    parallel_selector_html = plot_parallel_selector(filtered_dataset)

    with open(PLOTLY_COMBINED_HTML, 'w') as file:
        file.write(time_slider_html + acoustic_spectra_html + temperature_humidity_html + parallel_selector_html)
        logging.info(f"HTML file {PLOTLY_COMBINED_HTML} created")
    webbrowser.open(Path(PLOTLY_COMBINED_HTML).absolute().as_uri())
