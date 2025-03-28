import time
import logging
import webbrowser
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
    if len(filtered_ds['datetime'].values) == 0:
        logging.warning(f"No data found for sensor {sensor_id}")
        return None, None
    first_dt = min(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    last_dt = max(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    logging.info(f"Sensor {sensor_id}: first_dt={first_dt}, last_dt={last_dt}")
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

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    
    sensors = [116, 46, 21, 20]
    start = HELSINKI_4DAYS_AGO
    end = HELSINKI_NOW
  
    csv_files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
    )
    dataset = load_dataset(csv_files)
    logging.info(f"Loaded dataset with sensors: {np.unique(dataset.sensor)}")
  
    # Code section bellow will check if required HTML pieces very already created.
    # It will skip a lot of time on the seconds run because it does not try to recreate existing HTML pieces.
    if all([
        os.path.exists(TIME_SLIDER_HTML),
        os.path.exists(ACOUSTIC_SPECTRA_HTML),
    ]):
        with open(ACOUSTIC_SPECTRA_HTML, "r") as f:
            acoustic_spectra_html = f.read()
        with open(TIME_SLIDER_HTML, "r") as f:
            time_slider_html = f.read()
    else:
        acoustic_spectra_html = plot_acoustic_spectra(dataset, start, end)
        time_slider_html = plot_time_slider(dataset)
        _ = plot_similarity(dataset, start, end, OUTPUT_DIR)
        _ = plot_gaussians(dataset, start, end, OUTPUT_DIR)
        _ = plot_evolution(dataset, start, end, OUTPUT_DIR)

    with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
        acoustic_spectra_info = f.read()
    with open(SIMILARITY_INFO, "r") as f:
        similarity_info = f.read()

    # Prepare image paths for each sensor
    image_paths = {}
    for sensor in sensors:
        first_dt, last_dt = get_sensor_datetimes(dataset, sensor)
        if first_dt is not None and last_dt is not None:
            image_paths[sensor] = {
                'gauss': create_artifact_pathname('gauss', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
                'evolution': create_artifact_pathname('evolution', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
                'similarity': create_artifact_pathname('similarity', OUTPUT_DIR, sensor, first_dt, last_dt, 'png'),
            }
            logging.info(f"Created paths for sensor {sensor}: {image_paths[sensor]}")

    html_data = {
        "acoustic_spectra_plot": acoustic_spectra_html,
        "acoustic_spectra_info": acoustic_spectra_info,
        "time_slider_plot": time_slider_html,
        "similarity_info": similarity_info,
        "sensors": sensors,
        "OUTPUT_DIR": OUTPUT_DIR,
        "image_paths": image_paths,
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
