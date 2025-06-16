from pathlib import Path
import webbrowser
import logging
import numpy as np
import os
import json

from constants import *
from preprocessing import download_csv_if_needed, load_dataset, get_info_for_each_sensor
from plotly_plots import plot_acoustic_spectra, plot_time_slider
from gauss_plot import plot_averaged_and_individual_spectra
from evolution_plot import plot_evolution
from similarity_plot import plot_similarity
from peak_relations_plot import plot_peak_relations
from html_templating import create_html, get_sensor_datetimes

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("log.txt", mode="w"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    sensors = [109, 116, 117] 

    start = HELSINKI_2DAYS_AGO
    end = HELSINKI_NOW 
    # sensors = [116, 20]
    # start = HELSINKI_NOW - pd.Timedelta(hours=6)
    # end = HELSINKI_NOW 

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
    _ = plot_evolution(dataset, start, end, OUTPUT_DIR)
    _ = plot_peak_relations(dataset, sensors, start, end, OUTPUT_DIR)
    plot_averaged_and_individual_spectra(sensors, start, end)

    with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
        acoustic_spectra_info = f.read()
    with open(SIMILARITY_INFO, "r") as f:
        similarity_info = f.read()

    image_paths = {}
    for sensor in sensors:
        first_dt, last_dt = get_sensor_datetimes(filtered_dataset, sensor)
        image_paths[sensor] = {
            'gauss_averaged': create_artifact_pathname('gauss', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
            'gauss_individual': [create_artifact_pathname('gauss', OUTPUT_DIR, sensor, dt.astimezone(HELSINKI_TZ), dt.astimezone(HELSINKI_TZ), 'png') 
                               for dt in filtered_dataset.where(filtered_dataset.sensor == sensor, drop=True)['datetime'].values],
            'evolution': create_artifact_pathname('evolution', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
            'similarity': create_artifact_pathname('similarity', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
            'peak_relations': create_artifact_pathname('peak_relations', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
        }
        logging.info(f"Created paths for sensor {sensor}: {image_paths[sensor]}")
            
        for plot_type, path in image_paths[sensor].items():
            if isinstance(path, list):
                for p in path:
                    if os.path.exists(p):
                        logging.info(f"File exists: {p}")
                    else:
                        logging.warning(f"File does not exist: {p}")
            else:
                if os.path.exists(path):
                    logging.info(f"File exists: {path}")
                else:
                    logging.warning(f"File does not exist: {path}")

    with open(IMAGE_PATHS_JSON, 'w') as f:
        json.dump(image_paths, f, indent=2)
    logging.info(f"Saved image paths to {IMAGE_PATHS_JSON}")

    html_data = {
        "acoustic_spectra_plot": acoustic_spectra_html,
        "acoustic_spectra_info": acoustic_spectra_info,
        "time_slider_plot": time_slider_html,
        "similarity_info": similarity_info,
        "sensors": sensors,
        "OUTPUT_DIR": OUTPUT_DIR,
        "datapoints_info": get_info_for_each_sensor(filtered_dataset, start, end),
        "image_paths": image_paths,
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
