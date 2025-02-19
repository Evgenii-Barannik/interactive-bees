from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd
import os 

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

UTC_TZ = ZoneInfo('UTC')
HELSINKI_TZ = ZoneInfo('Europe/Helsinki')
HELSINKI_NOW = datetime.now(HELSINKI_TZ)
HELSINKI_4DAYS_AGO = HELSINKI_NOW - pd.Timedelta(days=4)
HELSINKI_24HOURS_AGO = HELSINKI_NOW - pd.Timedelta(hours=24)
HELSINKI_2DAYS_AGO = HELSINKI_NOW - pd.Timedelta(hours=48)
