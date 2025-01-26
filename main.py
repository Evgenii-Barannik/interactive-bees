import pandas as pd
import logging
import webbrowser
import os
from zoneinfo import ZoneInfo
from datetime import datetime
from pathlib import Path

from mpl_plots import *
from plotly_plots import * 
from html_templating import *
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

    averaged_plot_html, averaged_legend = create_averaged_plot_html(dataset, helsinki_days_ago, helsinki_now)
    phase_plot_html, phase_legend = create_temp_humidity_plot_html(dataset, dataset["sensor"].values, helsinki_24h_ago, helsinki_now)
    _, distance_legend = plot_correlations(dataset, helsinki_days_ago, helsinki_now, plots_dir)

    html_data = {
        "phase_plot_html": phase_plot_html,
        "averaged_plot_html": averaged_plot_html,
        "averaged_legend": averaged_legend,
        "phase_legend": phase_legend,
        "distance_legend": distance_legend,
    }

    html_path = render_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())

