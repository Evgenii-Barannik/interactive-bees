import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc, Output, Input
import logging
from pathlib import Path

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
    return fig

def plot_temperature_humidity(dataset, start, end):
    fig = go.Figure()
    all_sensors = dataset["sensor"].values
    colors = px.colors.sample_colorscale("Portland", len(all_sensors))
    last_datetime = max(dataset["datetime"].values).astimezone(HELSINKI_TZ)
    
    for i, sensor in enumerate(all_sensors):
        filtered_dataset = filter(dataset, sensor, start, end)
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
    
    sensors = dataset["sensor"].values

    info_datapoints = get_info_about_all_datapoints(dataset)
    with open(f'{OUTPUT_DIR}/{ACOUSTIC_SPECTRA_INFO}') as f:
        info_acoustic_spectra = f.read()
    with open(f'{OUTPUT_DIR}/{TEMPERATURE_HUMIDIY_INFO}') as f:
        info_th_diagram = f.read()
    with open(f'{OUTPUT_DIR}/{SIMILARITY_INFO}') as f:
        info_distance = f.read()

    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='spectrum-plot',
                className='plot-outer-container',
                config={
                    'scrollZoom': False,
                    'doubleClick': 'reset+autosize',
                    'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                }
            ),
        ], className='plot-outer-container'
        ),
        html.Div(id='time-slider-info'), # Will be updated by slider 
        dcc.RangeSlider( # Slider
            id='time-slider',
            min=0,
            max=len(time_positions_for_slider)-1,
            value=[range_start_idx, range_end_idx],
            marks={
                i: {'label': time_positions_for_slider[i].strftime('%d.%m'), 'style': {'white-space': 'nowrap'}}
                for i in range(0, len(time_positions_for_slider), 12)
            },
            step=1,
            allowCross=True
        ),
        html.Div([
            dcc.Graph(
                id='TH-plot',
                className='plot-outer-container',
                config={
                    'scrollZoom': False,
                    'doubleClick': 'reset+autosize',
                    'modeBarButtonsToRemove': ['lasso2d']
                }
            ),
        ],  className='plot-outer-container'
        ),
        
        # Distance Plot Section
        html.Div([
            html.Div([
                html.Div('Beehive acoustic similarity', className='plot-title'),
                dcc.Dropdown(
                    id='sensor-image-selector',
                    options=[
                        {'label': f'Sensor {sensor}', 'value': str(sensor)}
                        for sensor in sensors
                    ],
                    value=str(sensors[0]),
                    style={'width': '200px'}
                )
            ],
                     style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px auto'}
                     ),
                html.Img(
                    id='distance-plot-image',
                    className='image-container'
                )
        ]),
        
        # All Info Sections at the bottom
        html.Details([
            html.Summary('More info about acoustic spectra'),
            html.Pre(info_acoustic_spectra, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('More info about temperature-humidity diagram'),
            html.Pre(info_th_diagram, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('More info about distance plots'),
            html.Pre(info_distance, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('More info about datapoints'),
            html.Pre(info_datapoints, style={'white-space': 'pre-wrap'})
        ]),
        ],
        className='main-container')
    
    @app.callback(
        Output('time-slider-info', 'children'),
        Input('time-slider', 'value'),
    )
    def update_time_info(time_range):
        t0 = time_positions_for_slider[time_range[0]]
        t1 = time_positions_for_slider[time_range[1]]
        return html.Div([
            html.Div(f'Time range selected:'),
            html.Div(f'From: {t0}'),
            html.Div(f'To:\u00A0\u00A0\u00A0{t1}')
        ], style={'font-family': 'monospace'})

    @app.callback(
        Output('spectrum-plot', 'figure'),
        Input('time-slider', 'value')
    )
    def update_spectra(time_range):
        t0 = time_positions_for_slider[int(time_range[0])]
        t1 = time_positions_for_slider[int(time_range[1])]
        logging.info(f"Loading acoustic spectra for timerange:\n   FROM: {t0}\n   TO:   {t1}")
        return plot_acoustic_spectra(dataset, t0, t1)

    @app.callback(
        Output('TH-plot', 'figure'),
        Input('time-slider', 'value')
    )
    def update_TH_plot(time_range):
        t0 = time_positions_for_slider[int(time_range[0])]
        t1 = time_positions_for_slider[int(time_range[1])]
        logging.info(f"Loading temperature-humidity diagram for timerange:\n   FROM: {t0}\n   TO:   {t1}")
        return plot_temperature_humidity(dataset, t0, t1)

    @app.callback(
        Output('distance-plot-image', 'src'),
        Input('sensor-image-selector', 'value')
    )
    def update_image(sensor):
        if not sensor:
            sensor = str(sensors[0])
        image_path = Path(OUTPUT_DIR) / f'similarity-measures-sensor-{sensor}.png'
        if image_path.exists():
            logging.info(f"Loading existing image: {image_path}")
            return str(image_path)
        else:
            logging.info(f"Image does not exist: {image_path}")
            return str(image_path)

    logging.info(f"Dash App was created!")
    return app

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
    
    mpl_plot_pathnames = [(Path(OUTPUT_DIR) / f'similarity-measures-sensor-{sensor}.png') for sensor in sensors]
    mpl_plots_exist = all([p.exists() for p in mpl_plot_pathnames])
    if not mpl_plots_exist:
        correlations = plot_similarity(dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW, OUTPUT_DIR)
    else:
        logging.info(f"All similarity plots already exist.")
    
    app = create_app(dataset)
    app.run_server(debug=True)
