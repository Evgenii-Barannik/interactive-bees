from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd
import os 
import matplotlib as mpl
# import git
# repo = git.Repo(search_parent_directories=True)
# sha = repo.head.object.hexsha
# GIT_COMMIT_SHA = repo.git.rev_parse(sha, short=7)
ARTIFACT_NAME_BLUEPRINT = "{plotname}_{sensors}_from_{first_datetime}_to_{last_datetime}.{extension}"

def create_artifact_pathname(plotname, output_path, sensor_id, first_datetime, last_datetime, extension):
    pathname = os.path.join(output_path, ARTIFACT_NAME_BLUEPRINT.format(
        plotname = plotname,
        sensors = sensor_id,
        first_datetime = first_datetime.strftime("%Y-%m-%d-%H-%M-%S%z"),
        last_datetime =  last_datetime.strftime("%Y-%m-%d-%H-%M-%S%z"),
        extension = extension
    ))
    return pathname
    

OUTPUT_DIR = "assets"
DATA_DIR = "data"

ACOUSTIC_SPECTRA_HTML = os.path.join(OUTPUT_DIR, "acoustic_spectra_plot.html")
TIME_SLIDER_HTML = os.path.join(OUTPUT_DIR, "time_slider_plot.html")
TEMPERATURE_HUMIDIY_HTML = os.path.join(OUTPUT_DIR, "temperature_humidity_plot.html")
PARALLEL_SELECTOR_HTML = os.path.join(OUTPUT_DIR, "parallel_selector_plot.html")
PLOTLY_COMBINED_HTML = os.path.join(OUTPUT_DIR, "plots.html")

TEMPERATURE_HUMIDIY_INFO = os.path.join(OUTPUT_DIR, "temperature_humidity_info.txt")
SIMILARITY_INFO = os.path.join(OUTPUT_DIR, "similarity_info.txt")
ACOUSTIC_SPECTRA_INFO = os.path.join(OUTPUT_DIR, "acoustic_spectra_info.txt")
IMAGE_PATHS_JSON = os.path.join(OUTPUT_DIR, "image_paths.json")

UTC_TZ = ZoneInfo('UTC')
HELSINKI_TZ = ZoneInfo('Europe/Helsinki')
HELSINKI_NOW = datetime.now(HELSINKI_TZ)
HELSINKI_4DAYS_AGO = HELSINKI_NOW - pd.Timedelta(days=4)
HELSINKI_24HOURS_AGO = HELSINKI_NOW - pd.Timedelta(hours=24)
HELSINKI_2DAYS_AGO = HELSINKI_NOW - pd.Timedelta(hours=48)

COLORMAP_FOR_GAUSSIANS = mpl.colormaps['turbo_r']
FITTING_WINDOW_MIN = 60 # Hz    
FITTING_WINDOW_MAX = 650 # Hz
NORMALIZATION_LIMIT = 60 # Hz. THE SAME VALUE MUST BE USED IN JS CODE.
FITTING_MODEL = [
        {
            'type': 'background',
            'slope_guess': 0,
            'intercept_guess': 10,
        },
        {
            'type': 'peak', # A
            'center_range': (70, 155),
            'amplitude_guess': 70,
            'fwhm_guess': 30,
        },
        {  
            'type': 'peak', # B
            'center_range': (175, 220),
            'amplitude_guess': 40,
            'fwhm_guess': 50,
        },
        {  
            'type': 'peak', # C
            'center_range': (225, 275),
            'amplitude_guess': 40,
            'fwhm_guess': 50,
        },
        {  
            'type': 'peak', # D
            'center_range': (300, 350),
            'amplitude_guess': 40,
            'fwhm_guess': 50,
            'fwhm_max': 70
        },
        {  
            'type': 'peak', # E
            'center_range': (370, 440),
            'amplitude_guess': 20,
            'fwhm_guess': 30,
            'fwhm_max': 70
        },
        {  
            'type': 'peak', # F
            'center_range': (450, 500),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak', # G 
            'center_range': (500, 520),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak', # H
            'center_range': (550, 650),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        }
    ]
