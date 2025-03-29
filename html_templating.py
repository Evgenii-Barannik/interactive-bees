import time
import logging
import webbrowser
import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from constants import *
from plotly_plots import plot_time_slider, plot_acoustic_spectra
from gauss_plot import plot_gaussians
from evolution_plot import plot_evolution
from similarity_plot import plot_similarity
from preprocessing import download_csv_if_needed, load_dataset

def get_sensor_datetimes(dataset, sensor_id):
    filtered_ds = dataset.where(dataset.sensor == sensor_id, drop=True)
    assert len(filtered_ds['datetime'].values) != 0
    first_dt = min(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    last_dt = max(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    return first_dt, last_dt

def create_html(html_data):
    html_data["version"] = int(time.time()) 
    env = Environment(loader=FileSystemLoader(OUTPUT_DIR))
    template = env.get_template('template.html')
    html_filename = "index.html"
    rendered_html = template.render(html_data)
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    logging.info(f"HTML file created: {html_filename}")
    return html_filename

# if __name__ == "__main__":
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     handlers=[
    #         logging.StreamHandler()  # Log to console
    #     ]
    # )
    #
    # # Load image paths and get sensors from JSON first
    # assert os.path.exists(IMAGE_PATHS_JSON), f"Image paths JSON file not found at {IMAGE_PATHS_JSON}"
    # with open(IMAGE_PATHS_JSON, 'r') as f:
    #     image_paths = json.load(f)
    # logging.info(f"Loaded image paths from {IMAGE_PATHS_JSON}")
    #
    # # Get sensors from JSON and convert to integers for data processing
    # sensors = sorted([int(sensor) for sensor in image_paths.keys()])
    # logging.info(f"Using sensors from JSON: {sensors}")
    #
    # start = HELSINKI_4DAYS_AGO
    # end = HELSINKI_NOW
    #
    # csv_files = download_csv_if_needed(
    #         sensors,
    #         start.astimezone(UTC_TZ),
    #         end.astimezone(UTC_TZ),
    #         DATA_DIR
    # )
    # dataset = load_dataset(csv_files)
    #
    # # Code section bellow will check if required HTML pieces very already created.
    # # It will skip a lot of time on the seconds run because it does not try to recreate existing HTML pieces.
    # if all([
    #     os.path.exists(TIME_SLIDER_HTML),
    #     os.path.exists(ACOUSTIC_SPECTRA_HTML),
    # ]):
    #     with open(ACOUSTIC_SPECTRA_HTML, "r") as f:
    #         acoustic_spectra_html = f.read()
    #     with open(TIME_SLIDER_HTML, "r") as f:
    #         time_slider_html = f.read()
    # else:
    #     acoustic_spectra_html = plot_acoustic_spectra(dataset, start, end)
    #     time_slider_html = plot_time_slider(dataset)
    #     _ = plot_similarity(dataset, start, end, OUTPUT_DIR)
    #     _ = plot_gaussians(dataset, start, end, OUTPUT_DIR)
    #     _ = plot_evolution(dataset, start, end, OUTPUT_DIR)
    #
    # with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
    #     acoustic_spectra_info = f.read()
    # with open(SIMILARITY_INFO, "r") as f:
    #     similarity_info = f.read()
    #
    # # Verify all required images exist
    # for sensor, paths in image_paths.items():
    #     for plot_type, path in paths.items():
    #         assert os.path.exists(path), f"Required image not found: {path} for sensor {sensor}, plot type {plot_type}"
    #
    # # Convert numpy.int64 sensors to strings for template
    # template_sensors = [str(sensor) for sensor in sensors]
    #
    # html_data = {
    #     "acoustic_spectra_plot": acoustic_spectra_html,
    #     "acoustic_spectra_info": acoustic_spectra_info,
    #     "time_slider_plot": time_slider_html,
    #     "similarity_info": similarity_info,
    #     "sensors": template_sensors,
    #     "OUTPUT_DIR": OUTPUT_DIR,
    #     "image_paths": image_paths,
    # }
    #
    # html_path = create_html(html_data)
    # webbrowser.open(Path(html_path).absolute().as_uri())
