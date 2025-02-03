from pathlib import Path
from dash import Dash, html, dcc, Output, Input
import logging

from preprocessing import *
from mpl_plots import *
from plotly_plots_Dash import * 
from constants import *

def create_app(dataset, global_start, global_end):
    app = Dash(__name__, assets_folder='assets')
    time_positions_for_slider = pd.date_range(
            start=global_start.astimezone(HELSINKI_TZ).replace(minute=0, second=0, microsecond=0),  
            end=global_end.astimezone(HELSINKI_TZ).replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1),
            freq='2h'  
    )
    range_start_idx = 0
    range_end_idx = len(time_positions_for_slider) - 1
    
    sensors = dataset["sensor"].values

    info_datapoints = get_info_about_all_datapoints(dataset, global_start, global_end)
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
    def update_time_info(sliders_positions):
        t0 = time_positions_for_slider[sliders_positions[0]]
        t1 = time_positions_for_slider[sliders_positions[1]]
        return html.Div([
            html.Div(f'Time range selected:'),
            html.Div(f'From: {t0}'),
            html.Div(f'To:\u00A0\u00A0\u00A0{t1}')
        ], style={'font-family': 'monospace'})

    @app.callback(
        Output('spectrum-plot', 'figure'),
        Input('time-slider', 'value')
    )
    def update_spectra(sliders_positions):
        t0 = time_positions_for_slider[int(sliders_positions[0])]
        t1 = time_positions_for_slider[int(sliders_positions[1])]
        logging.info(f"Loading acoustic spectra for timerange:\n   FROM: {t0}\n   TO:   {t1}")
        return plot_acoustic_spectra(dataset, t0, t1)

    @app.callback(
        Output('TH-plot', 'figure'),
        Input('time-slider', 'value')
    )
    def update_TH_plot(sliders_positions):
        t0 = time_positions_for_slider[int(sliders_positions[0])]
        t1 = time_positions_for_slider[int(sliders_positions[1])]
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
            logging.FileHandler("log.txt", mode="w"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    
    sensors = [20, 21, 46, 109]
    global_start = HELSINKI_4DAYS_AGO
    global_end = HELSINKI_NOW
    csv_files = download_csv_if_needed(
            sensors,
            global_start.astimezone(UTC_TZ),
            global_end.astimezone(UTC_TZ),
            DATA_DIR
    ) 
    dataset = load_dataset(csv_files)

    # mpl_plot_pathnames = [(Path(OUTPUT_DIR) / f'similarity-measures-sensor-{sensor}.png') for sensor in sensors]
    # mpl_plots_exist = all([p.exists() for p in mpl_plot_pathnames])
    # if not mpl_plots_exist: # Local images can be out of date, but much faster repeats
    correlations = plot_similarity(dataset, global_start, global_end, OUTPUT_DIR)
    # else:
    #     logging.info(f"All similarity plots already exist.")
    
    app = create_app(dataset, global_start, global_end)
    app.run_server(debug=True)
