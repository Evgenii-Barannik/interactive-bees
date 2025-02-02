from zoneinfo import ZoneInfo
from datetime import datetime
import pandas as pd

OUTPUT_DIR = "assets"
DATA_DIR = "data"
ACOUSTIC_SPECTRA_INFO = "acoustic_spectra_info.txt"
TEMPERATURE_HUMIDIY_INFO = "temperature_humidity_info.txt"
SIMILARITY_INFO = "similarity_info.txt"

UTC_TZ = ZoneInfo('UTC')
HELSINKI_TZ = ZoneInfo('Europe/Helsinki')
HELSINKI_NOW = datetime.now(HELSINKI_TZ)
HELSINKI_4DAYS_AGO = HELSINKI_NOW - pd.Timedelta(days=4)
HELSINKI_24HOURS_AGO = HELSINKI_NOW - pd.Timedelta(hours=24)
