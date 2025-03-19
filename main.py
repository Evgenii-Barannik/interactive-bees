from pathlib import Path
import webbrowser
import logging

from constants import *
from preprocessing import download_csv_if_needed, load_dataset, get_info_for_each_sensor
from plotly_plots import plot_acoustic_spectra, plot_time_slider, plot_parallel_selector
from gauss_plot import plot_gaussians
from evolution_plot import plot_evolution
from similarity_plot import plot_similarity
from html_templating import create_html

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("log.txt", mode="w"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    sensors = [46, 101, 116]
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

    time_slider_html = plot_time_slider(filtered_dataset)
    acoustic_spectra_html = plot_acoustic_spectra(filtered_dataset, start, end)
    _ = plot_similarity(dataset, start, end, OUTPUT_DIR)
    _ = plot_gaussians(dataset, start, end, OUTPUT_DIR)
    _ = plot_evolution(dataset, start, end, OUTPUT_DIR)

    with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
        acoustic_spectra_info = f.read()
    with open(SIMILARITY_INFO, "r") as f:
        similarity_info = f.read()

    html_data = {
        "acoustic_spectra_plot" : acoustic_spectra_html,
        "acoustic_spectra_info" : acoustic_spectra_info,
        "time_slider_plot": time_slider_html,
        "similarity_info": similarity_info,
        "sensors": sensors,
        "OUTPUT_DIR": OUTPUT_DIR,
        "datapoints_info": get_info_for_each_sensor(filtered_dataset, start, end),
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
