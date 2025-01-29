import plotly.graph_objects as go
import numpy as np
import os
import webbrowser
from pathlib import Path
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc, Output, Input

from preprocessing import * 
from mpl_plots import *
from constants import *

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

def plot_acoustic_spectra(dataset, output_path):
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    
    app = Dash(__name__)
    
    # Get time range
    times = dataset["datetime"].values
    time_steps = pd.date_range(start=min(times), end=max(times), freq='6h')
    default_start_idx = np.argmin(np.abs(time_steps - HELSINKI_4DAYS_AGO))
    default_end_idx = len(time_steps) - 1
    
    app.layout = html.Div([
        dcc.Graph(id='spectrum-plot'),
        dcc.RangeSlider(
            id='time-slider',
            min=0,
            max=len(time_steps)-1,
            value=[default_start_idx, default_end_idx],
            marks={
                i: time_steps[i].strftime('%d-%m %H:%M')
                for i in range(0, len(time_steps), 4)  # Show every 4th mark
            },
            step=1
        )
    ])
    
    @app.callback(
        Output('spectrum-plot', 'figure'),
        Input('time-slider', 'value')
    )
    def update_figure(time_range):
        start_time = time_steps[int(time_range[0])]
        end_time = time_steps[int(time_range[1])]
        
        fig = go.Figure()
        all_sensors = dataset["sensor"].values
        colors = px.colors.sample_colorscale("Portland", len(all_sensors))
        
        for i, sensor in enumerate(all_sensors):
            filtered_dataset = filter(dataset, sensor, start_time, end_time)
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
                name=f'Sensor {sensor}',
                marker_color=colors[i],
                opacity=0.7,
                hovertemplate='%{y:.1f}%, %{x:.1f} Hz'
            ))
        
        fig.update_layout(
            title='Beehive acoustic spectra',
            xaxis_title='Frequency, Hz',
            yaxis_title='Relative Amplitude, %',
            xaxis_range=[0, 850],
            yaxis_range=[0, 100],
            hovermode='closest',
            legend=dict(
                yanchor="top",
                xanchor="right",
                bgcolor='rgba(255, 255, 255, 0.5)'
            )
        )
        
        return fig
    
    if output_path:
        initial_fig = update_figure([default_start_idx, default_end_idx])
        rendered_html = initial_fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
        os.makedirs(output_path, exist_ok=True)
        with open(ACOUSTIC_SPECTRA_PLOT_PATHNAME, 'w') as file:
            file.write(rendered_html)
        return rendered_html, ""
    
    return app

def plot_temperature_humidity(dataset, sensors, start, end, output_path):
    fig = go.Figure()
    text_for_html_annotation = f"""
    Temperature-Humidity phase plot.<br>
    Points with more vivid colors are more recent.<br><br>
    """
    colors = px.colors.sample_colorscale("Portland", len(sensors))
    last_datetime = max(dataset["datetime"].values).astimezone(HELSINKI_TZ)
    
    for i, sensor in enumerate(sensors):
        filtered_dataset = filter(dataset, sensor, start, end)
        
        temperatures = filtered_dataset['temperature'].values
        humidities = filtered_dataset['humidity'].values
        times = filtered_dataset['datetime'].values
        
        floating_hours_time_ago = [(last_datetime - t).total_seconds()/3600 for t in times]
        max_ago = max(floating_hours_time_ago)
        opacities = [0.2 + 0.8 * (1 - t/max_ago) for t in floating_hours_time_ago]
        text_for_html_annotation = text_for_html_annotation + "For sensor {}:<br>{}<br>".format(sensor, get_info_about_datapoints(filtered_dataset)).replace('\n', '<br>')
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
                    ['{}'.format(t.astimezone(HELSINKI_TZ)) for t in times],
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

    # Plot and info are both saved and returned
    rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
    os.makedirs(output_path, exist_ok=True)
    with open(TEMPERATURE_HUMIDIY_PLOT_PATHNAME, 'w') as file:
        file.write(rendered_html)
    with open(TEMPERATURE_HUMIDIY_INFO_PATHNAME, 'w') as file:
        file.write(text_for_html_annotation)
    
    logging.info(f"HTML file {TEMPERATURE_HUMIDIY_PLOT_PATHNAME} was created!")
    return rendered_html, text_for_html_annotation

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    sensors = [20, 21, 46, 109]
    csv_files = download_csv_if_needed(sensors, HELSINKI_4DAYS_AGO, HELSINKI_NOW, DATA_DIR) 
    dataset = load_dataset(csv_files)
    
    # Run the interactive Dash app for acoustic spectra
    app = plot_acoustic_spectra(dataset, None)
    app.run_server(debug=True)
