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

def get_info_for_acoustic_spectra_plot(dataset):
    datetimes = dataset["datetime"].values
    info_text = "Each shown spectrum is created by averaging spectra from multiple datapoints and then normalizing the result.\n\n"
    for sensor in dataset["sensor"].values:
        filtered_dataset = filter(dataset, sensor, min(datetimes), max(datetimes)) # Gives all points for specific sensor
        if filtered_dataset['datetime'].shape[0] > 0:
            info_text += f"**For sensor {sensor}:**\n{get_info_about_datapoints(filtered_dataset)}\n\n"
    return info_text

def plot_acoustic_spectra(dataset):
    app = Dash(__name__, assets_folder='assets')
    
    times = dataset["datetime"].values
    time_steps = pd.date_range(start=min(times), end=max(times), freq='2h')
    range_start_idx = np.argmin(np.abs(time_steps - HELSINKI_4DAYS_AGO))
    range_end_idx = len(time_steps) - 1
    
    info = get_info_for_acoustic_spectra_plot(dataset)
    app.layout = html.Div([
        ### PLOT AND CONTROL:
        html.Div([
            dcc.Graph(
                id='spectrum-plot',
                className='plot-container',
                config={
                    'scrollZoom': False,
                    'doubleClick': 'reset+autosize',
                    'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                    }
                ),
            html.Div(id='spectrum-slider-info'), # Will be updated 
            dcc.RangeSlider(
                id='spectrum-slider',
                min=0,
                max=len(time_steps)-1,
                value=[range_start_idx, range_end_idx],
                marks={
                    i: {'label': time_steps[i].strftime('%d.%m'), 'style': {'white-space': 'nowrap'}}
                    for i in range(0, len(time_steps), 12)
                    },
                step=1,
                pushable=1,
                allowCross=False
                ),
            ],
            className='plot-and-control-container',
            ),
        ### PLOT AND CONTROL END ^
        html.Details([
            html.Summary('More info'),
            html.Pre(info, style={'white-space': 'pre-wrap'})
        ],
         className='details-container',
        )
    ], className='main-container')
    
    
    @app.callback(
        Output('spectrum-slider-info', 'children'),
        Input('spectrum-slider', 'value'),
    )
    def update_output(value):
        ts0 = time_steps[value[0]]
        ts1 = time_steps[value[1]]
        return html.Div([
            html.Div(f'Time range used for averaging:'),
            html.Div(f'From: {ts0}'),
            html.Div(f'To:\u00A0\u00A0\u00A0{ts1}')
        ], style={'font-family': 'monospace'})


    @app.callback(
        Output('spectrum-plot', 'figure'),
        Input('spectrum-slider', 'value')
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
    return app

# def plot_temperature_humidity(dataset, sensors, start, end, output_path):
#     fig = go.Figure()
#     text_for_html_annotation = f"""
#     Temperature-Humidity phase plot.<br>
#     Points with more vivid colors are more recent.<br><br>
#     """
#     colors = px.colors.sample_colorscale("Portland", len(sensors))
#     last_datetime = max(dataset["datetime"].values).astimezone(HELSINKI_TZ)
    
#     for i, sensor in enumerate(sensors):
#         filtered_dataset = filter(dataset, sensor, start, end)
        
#         temperatures = filtered_dataset['temperature'].values
#         humidities = filtered_dataset['humidity'].values
#         times = filtered_dataset['datetime'].values
        
#         floating_hours_time_ago = [(last_datetime - t).total_seconds()/3600 for t in times]
#         max_ago = max(floating_hours_time_ago)
#         opacities = [0.2 + 0.8 * (1 - t/max_ago) for t in floating_hours_time_ago]
#         text_for_html_annotation = text_for_html_annotation + "For sensor {}:<br>{}<br>".format(sensor, get_info_about_datapoints(filtered_dataset)).replace('\n', '<br>')
#         fig.add_trace(
#             go.Scatter(
#                 x=temperatures,
#                 y=humidities,
#             mode='markers',
#                 marker=dict(
#                     size=8,
#                     color=colors[i],
#                     opacity=opacities
#                 ),
#             name=f'Sensor {sensor}',
#                 hovertemplate=(
#                     '%{fullData.name}<br>'
#                     'Temperature: %{x:.1f}°C<br>'
#                     'Humidity: %{y:.1f}%<br>'
#                     'DateTime: %{customdata[0]}<br>'
#                     'Ago from most recent datapoint: %{customdata[1]:.1f} h<extra></extra>'
#                 ),
#                 customdata=list(zip(
#                     ['{}'.format(t.astimezone(HELSINKI_TZ)) for t in times],
#                     floating_hours_time_ago,
#                 ))
#             )
#         )
    
#     fig.update_layout(
#         title=dict(
#             text='Beehive temperature and humidity',
#         ),
#         xaxis_title='Temperature, °C',
#         yaxis_title='Relative Humidity, %',
#         hovermode='closest',
#         legend=dict(
#             yanchor="top",     
#             xanchor="right",
#             bgcolor='rgba(255, 255, 255, 0.5)',
#         ),
#         margin=dict(l=25, r=25, t=75, b=0, pad=0)
#     )

#     # Plot and info are both saved and returned
#     rendered_html = fig.to_html(config={'responsive': True}, include_plotlyjs=True, full_html=False)
#     os.makedirs(output_path, exist_ok=True)
#     with open(TEMPERATURE_HUMIDIY_PLOT_PATHNAME, 'w') as file:
#         file.write(rendered_html)
#     with open(TEMPERATURE_HUMIDIY_INFO_PATHNAME, 'w') as file:
#         file.write(text_for_html_annotation)
    
#     logging.info(f"HTML file {TEMPERATURE_HUMIDIY_PLOT_PATHNAME} was created!")
#     return rendered_html, text_for_html_annotation

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
    app = plot_acoustic_spectra(dataset)
    app.run_server(debug=True)
