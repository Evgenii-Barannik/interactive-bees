from pathlib import Path
import logging

from preprocessing import *
from mpl_plots import *
from constants import *
from html_templating import *

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
    start = HELSINKI_2DAYS_AGO
    end = HELSINKI_NOW
    csv_files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
    ) 
    
    dataset = load_dataset(csv_files)
    filtered_dataset = load_dataset(csv_files, True, start, end)
    acoustic_spectra_html = plot_acoustic_spectra(
            filtered_dataset, start, end
    )
    temperature_humidity_html = plot_temperature_humidity(
            filtered_dataset, start, end
    )
    time_slider_html = plot_time_slider(
            filtered_dataset, start, end
    )
    _ = plot_similarity(dataset, start, end, OUTPUT_DIR)

    with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
        acoustic_spectra_info = f.read()
    with open(TEMPERATURE_HUMIDIY_INFO, "r") as f:
        temperature_humidity_info = f.read()
    with open(SIMILARITY_INFO, "r") as f:
        similarity_info = f.read()

    html_data = {
        "acoustic_spectra_plot" : acoustic_spectra_html,
        "acoustic_spectra_info" : acoustic_spectra_info,
        "temperature_humidity_plot": temperature_humidity_html,
        "time_slider_plot": time_slider_html,
        "temperature_humidity_info": temperature_humidity_info,
        "similarity_info": similarity_info,
        "OUTPUT_DIR": OUTPUT_DIR,

        "datapoints_info": get_info_about_all_datapoints(filtered_dataset, start, end)
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
