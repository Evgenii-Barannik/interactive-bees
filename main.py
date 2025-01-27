import logging
import webbrowser
from pathlib import Path

from html_templating import create_html
from mpl_plots import *
from plotly_plots import * 
from preprocessing import *
from constants import *

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
    csv_files = download_csv_if_needed(sensors, HELSINKI_4DAYS_AGO, HELSINKI_NOW, DATA_DIR) 
    dataset = load_dataset(csv_files)
    acoustic_spectra_plot, acoustic_spectra_info = plot_acoustic_spectra(
            dataset,
            HELSINKI_4DAYS_AGO,
            HELSINKI_NOW,
            OUTPUT_DIR
    )
    temperature_humidity_plot, temperature_humidity_info = plot_temperature_humidity(
            dataset,
            dataset["sensor"].values,
            HELSINKI_24HOURS_AGO,
            HELSINKI_NOW,
            OUTPUT_DIR
    )
    _, similarity_info = plot_similarity(dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW, OUTPUT_DIR)

    html_data = {
        "acoustic_spectra_plot" : acoustic_spectra_plot,
        "acoustic_spectra_info" : acoustic_spectra_info,
        "temperature_humidity_plot": temperature_humidity_plot,
        "temperature_humidity_info": temperature_humidity_info,
        "similarity_info": similarity_info,
        "OUTPUT_DIR": OUTPUT_DIR
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
