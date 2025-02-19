from pathlib import Path
from dash import Dash, html, dcc, Output, Input
import logging

from preprocessing import *
from mpl_plots import *
from plotly_plots import * 
from constants import *

def create_app(dataset, global_start, global_end):
    app = Dash(__name__, assets_folder='assets')
    sensors = dataset["sensor"].values
    info_datapoints = get_info_about_all_datapoints(dataset, global_start, global_end)
    with open(f'{ACOUSTIC_SPECTRA_INFO}') as f:
        info_acoustic_spectra = f.read()
    with open(f'{TEMPERATURE_HUMIDIY_INFO}') as f:
        info_th_diagram = f.read()
    with open(f'{SIMILARITY_INFO}') as f:
        info_distance = f.read()

    start=global_start.astimezone(HELSINKI_TZ).replace(minute=0, second=0, microsecond=0)
    end=global_end.astimezone(HELSINKI_TZ).replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)

    app.layout = html.Div([
            dcc.Graph(
                id='acoustic-spectra-plot',
                className='plotly-container',
                # config={
                #     'scrollZoom': False,
                #     'doubleClick': 'reset+autosize',
                #     'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                # }
            ),
            dcc.Graph(
                id='time-selector-plot',
                className='plotly-container',
                # config={
                #     'scrollZoom': False,
                #     'doubleClick': 'reset+autosize',
                #     'modeBarButtonsToRemove': ['lasso2d', 'pan2d']
                # }
            ),
            # dcc.Graph(
            #     id='temperature-humidity-plot',
            #     className='plotly-container',
            #     config={
            #         'scrollZoom': False,
            #         'doubleClick': 'reset+autosize',
            #         'modeBarButtonsToRemove': ['lasso2d']
            #         }
            #     ),

    #     html.Div(id='time-slider-info'), # Will be updated by slider 
    #     dcc.RangeSlider( # Slider
    #         id='time-slider',
    #         min=0,
    #         max=len(time_positions_for_slider)-1,
    #         value=[range_start_idx, range_end_idx],
    #         marks={
    #             i: {'label': time_positions_for_slider[i].strftime('%d.%m'), 'style': {'white-space': 'nowrap'}}
    #             for i in range(0, len(time_positions_for_slider), 12)
    #         },
    #         step=1,
    #         allowCross=True
    #     ),
    #     html.Div([
    #     ],  className='plot-outer-container'
    #     ),
    #
        html.Div([
            html.H4('Acoustic similarity', style={'margin': 0}),
            dcc.Dropdown(
                id='sensor-selector',
                options=[
                    {'label': f'Sensor {sensor}', 'value': str(sensor) } for sensor in sensors
                ],
                value=str(sensors[0]),
                className='sensor-selector'
            ),
        ], className='header-and-selector'),

        html.Img(
            id='similarity-image',
            className='image-container'
        ),

        html.Details([
            html.Summary('About acoustic spectra'),
            html.Pre(info_acoustic_spectra, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('About temperature-humidity diagram'),
            html.Pre(info_th_diagram, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('About acoustic similarity diagrams'),
            html.Pre(info_distance, style={'white-space': 'pre-wrap'})
        ]),
        html.Details([
            html.Summary('About datapoints'),
            html.Pre(info_datapoints, style={'white-space': 'pre-wrap'})
        ]),

    ], className='main-container')
    #
    # @app.callback(
    #     Output('time-slider-info', 'children'),
    #     Input('time-slider', 'value'),
    # )
    # def update_time_info(sliders_positions):
    #     t0 = time_positions_for_slider[sliders_positions[0]]
    #     t1 = time_positions_for_slider[sliders_positions[1]]
    #     return html.Div([
    #         html.Div(f'Time range selected:'),
    #         html.Div(f'From: {t0}'),
    #         html.Div(f'To:\u00A0\u00A0\u00A0{t1}')
    #     ], style={'font-family': 'monospace'})

    @app.callback(
        Output('acoustic-spectra-plot', 'figure'),
        Input('time-selector-plot', 'value')
    )
    def update_acoustic_spectra_plot(sliders_positions):
        t0 = datetime_range[int(sliders_positions[0])]
        t1 = datetime_range[int(sliders_positions[1])]
        logging.info(f"Loading acoustic spectra for timerange:\n   FROM: {t0}\n   TO:   {t1}")
        return plot_acoustic_spectra(dataset, t0, t1, return_fig=True)


    # @app.callback(
    #     Output('temperature-humidity-plot', 'figure'),
    #     Input('time-slider', 'value')
    # )
    # def update_temperature_humidity_plot(sliders_positions):
    #     t0 = time_positions_for_slider[int(sliders_positions[0])]
    #     t1 = time_positions_for_slider[int(sliders_positions[1])]
    #     logging.info(f"Loading temperature-humidity diagram for timerange:\n   FROM: {t0}\n   TO:   {t1}")
    #     return plot_temperature_humidity(dataset, t0, t1)


    @app.callback(
        Output('similarity-image', 'src'),
        Input('sensor-selector', 'value')
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

    
    sensors = [109, 116]
    global_start = HELSINKI_4DAYS_AGO
    global_end = HELSINKI_NOW
    csv_files = download_csv_if_needed(
            sensors,
            global_start.astimezone(UTC_TZ),
            global_end.astimezone(UTC_TZ),
            DATA_DIR
    ) 
    dataset = load_dataset(csv_files)

    mpl_plot_pathnames = [(Path(OUTPUT_DIR) / f'similarity-measures-sensor-{sensor}.png') for sensor in sensors]
    mpl_plots_exist = all([p.exists() for p in mpl_plot_pathnames])
    if not mpl_plots_exist: # Local images can be out of date, but it makes rebuilds much faster
        correlations = plot_similarity(dataset, global_start, global_end, OUTPUT_DIR)
    else:
        logging.info(f"All similarity plots already exist.")
    
    app = create_app(dataset, global_start, global_end)
    webbrowser.open("http://127.0.0.1:8050/", new=1, autoraise=True)
    app.run_server(debug=True)
