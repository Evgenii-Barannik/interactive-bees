import plotly.graph_objects as go
import numpy as np
# import os
# import webbrowser
# from pathlib import Path
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

def get_info_for_acoustic_spectra(dataset):
    datetimes = dataset["datetime"].values
    info_text = "Each shown spectrum is created by averaging spectra from multiple datapoints and then normalizing the result.\n\n"
    for sensor in dataset["sensor"].values:
        filtered_dataset = filter(dataset, sensor, min(datetimes), max(datetimes)) # Gives all points for specific sensor
        if filtered_dataset['datetime'].shape[0] > 0:
            info_text += f"**For sensor {sensor}:**\n{get_info_about_datapoints(filtered_dataset)}\n\n"
    return info_text

def plot_acoustic_spectra(dataset, start_time, end_time):
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
    return fig

def create_app(dataset):
    app = Dash(__name__, assets_folder='assets')
    times = dataset["datetime"].values
    time_positions_for_slider = pd.date_range(
            start=min(times).astimezone(HELSINKI_TZ).replace(minute=0, second=0),  
            end=max(times).astimezone(HELSINKI_TZ).replace(minute=0, second=0) + pd.Timedelta(hours=1),
            freq='2h'  
    )
    range_start_idx = 0
    range_end_idx = len(time_positions_for_slider) - 1
    
    info_acoustic_spectra = get_info_for_acoustic_spectra(dataset)
    info_th_diagramm = get_info_for_th_diagramm(dataset)
    app.layout = html.Div([
        ### ACOUSTIC SPECTRUM PLOT ###
        html.Div([
            dcc.Graph( # Will be updated by slider 
                id='spectrum-plot',
                className='plot-container',
                config={
                    'scrollZoom': False,
                    'doubleClick': 'reset+autosize',
                    'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                    }
                ),
            html.Div(id='spectrum-slider-info'), # Will be updated by slider 
            dcc.RangeSlider( # Slider
                id='spectrum-slider',
                min=0,
                max=len(time_positions_for_slider)-1,
                value=[range_start_idx, range_end_idx],
                marks={
                    i: {'label': time_positions_for_slider[i].strftime('%d.%m'), 'style': {'white-space': 'nowrap'}}
                    for i in range(0, len(time_positions_for_slider), 12)
                    },
                step=1,
                pushable=1,
                allowCross=False
                ),
            ],
            className='plot-and-control-container',
            ),
        html.Details([
            html.Summary('More info'),
            html.Pre(info_acoustic_spectra, style={'white-space': 'pre-wrap'})
        ],
         className='details-container',
        ),
        ### TH DIAGRAMM PLOT ###
        html.Div([
            dcc.Graph( # Will be updated by slider 
                id='TH-plot',
                className='plot-container',
                config={
                    'scrollZoom': False,
                    'doubleClick': 'reset+autosize',
                    'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                    }
                ),
            html.Div(id='TH-slider-info'), # Will be updated by slider 
            dcc.RangeSlider( # Slider
                id='TH-slider',
                min=0,
                max=len(time_positions_for_slider)-1,
                value=[range_start_idx, range_end_idx],
                marks={
                    i: {'label': time_positions_for_slider[i].strftime('%d.%m'), 'style': {'white-space': 'nowrap'}}
                    for i in range(0, len(time_positions_for_slider), 12)
                    },
                step=1,
                pushable=1,
                allowCross=False
                ),
            ],
            className='plot-and-control-container',
            ),
        html.Details([
            html.Summary('More info'),
            html.Pre(info_th_diagramm, style={'white-space': 'pre-wrap'})
        ],
         className='details-container',
        ),
        ],
        className='main-container')
    
    
    @app.callback(
        Output('spectrum-slider-info', 'children'),
        Input('spectrum-slider', 'value'),
    )
    def update_spectrum_info(value):
        ts0 = time_positions_for_slider[value[0]]
        ts1 = time_positions_for_slider[value[1]]
        return html.Div([
            html.Div(f'Time range used for averaging:'),
            html.Div(f'From: {ts0}'),
            html.Div(f'To:\u00A0\u00A0\u00A0{ts1}')
        ], style={'font-family': 'monospace'})

    @app.callback(
        Output('spectrum-plot', 'figure'),
        Input('spectrum-slider', 'value')
    )
    def update_spectra(time_range):
        start_time = time_positions_for_slider[int(time_range[0])]
        end_time = time_positions_for_slider[int(time_range[1])]
        return plot_acoustic_spectra(dataset, start_time, end_time)

    @app.callback(
        Output('TH-plot', 'figure'),
        Input('TH-slider', 'value')
    )
    def update_TH_plot(time_range):
        start_time = time_positions_for_slider[int(time_range[0])]
        end_time = time_positions_for_slider[int(time_range[1])]
        return plot_temperature_humidity(dataset, start_time, end_time)

    @app.callback(
        Output('TH-slider-info', 'children'),
        Input('TH-slider', 'value'),
    )
    def update_TH_info(value):
        ts0 = time_positions_for_slider[value[0]]
        ts1 = time_positions_for_slider[value[1]]
        return html.Div([
            html.Div(f'Time range used for plotting:'),
            html.Div(f'From: {ts0}'),
            html.Div(f'To:\u00A0\u00A0\u00A0{ts1}')
        ], style={'font-family': 'monospace'})

    logging.info(f"Dash App was created!")
    return app

def get_info_for_th_diagramm(dataset):
    datetimes = dataset["datetime"].values
    info_text = "Temperature-Humidity phase plot. Points with more vivid colors are more recent.\n\n"
    for sensor in dataset["sensor"].values:
        filtered_dataset = filter(dataset, sensor, min(datetimes), max(datetimes)) # Gives all points for specific sensor
        if filtered_dataset['datetime'].shape[0] > 0:
            info_text += f"**For sensor {sensor}:**\n{get_info_about_datapoints(filtered_dataset)}\n\n"
    return info_text

def plot_temperature_humidity(dataset, start, end):
    fig = go.Figure()
    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    last_datetime = max(dataset["datetime"].values).astimezone(HELSINKI_TZ)
    
    for i, sensor in enumerate(all_sensors):
        filtered_dataset = filter(dataset, sensor, start, end)
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
    return fig

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
    
    app = create_app(dataset)
    app.run_server(debug=True)
