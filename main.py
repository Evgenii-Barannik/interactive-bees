import pandas as pd
import logging
import webbrowser
import os
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path

from correlations_plotting import *
from averaged_spectra_plotting import * 
from preprocessing import *

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("log.txt", mode="w"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=4)
    helsinki_24h_ago = helsinki_now - pd.Timedelta(days=1)

    data_dir = "data"
    plots_dir = "plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    sensors = [20, 21, 46, 109]
    csv_files = download_csv_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(csv_files)

    os.makedirs(plots_dir, exist_ok=True)
    html_pathname = create_html(
        dataset, 
        start_old=helsinki_days_ago,
        start_recent=helsinki_24h_ago,
        end=helsinki_now, 
        plot_output_path=plots_dir
    )
    index_path = Path(html_pathname).absolute().as_uri()
    webbrowser.open(index_path)
