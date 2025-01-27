import os
from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd

OUTPUT_DIR = "output"
DATA_DIR = "data"
ACOUSTIC_SPECTRA_PLOT_PATHNAME = os.path.join(OUTPUT_DIR, "acoustic_spectra_plot.html")
ACOUSTIC_SPECTRA_INFO_PATHNAME = os.path.join(OUTPUT_DIR, "acoustic_spectra_info.html")
TEMPERATURE_HUMIDIY_PLOT_PATHNAME = os.path.join(OUTPUT_DIR, "temperature_humidity_plot.html")
TEMPERATURE_HUMIDIY_INFO_PATHNAME = os.path.join(OUTPUT_DIR, "temperature_humidity_info.html")
SIMILARITY_INFO_PATHNAME = os.path.join(OUTPUT_DIR, "similarity_info.html")

PLOTLY_PLOTS_PATHNAME = os.path.join(OUTPUT_DIR, "plotly_plots.html")

HELSINKI_TZ = ZoneInfo('Europe/Helsinki')
HELSINKI_NOW = datetime.now(HELSINKI_TZ)
HELSINKI_4DAYS_AGO = HELSINKI_NOW - pd.Timedelta(days=4)
HELSINKI_24HOURS_AGO = HELSINKI_NOW - pd.Timedelta(hours=24)

