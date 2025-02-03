import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import logging

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
            hovertemplate='(%{y:.1f}%, %{x:.1f} Hz<extra></extra>)', #<extra></extra> removes sensor number(it is obvious anyway)
        ))
    
    fig.update_layout(
        title='Beehive acoustic spectra',
        xaxis_title='Frequency, Hz',
        yaxis_title='Relative Amplitude, %',
        xaxis_range=[0, 850],
        yaxis_range=[0, 100],
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
    logging.info(f"Acoustic spectra plotted.")
    return fig

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
    logging.info(f"Temperature-Humidity diagram plotted.")
    return fig
