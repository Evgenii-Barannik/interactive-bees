import time
import logging
import webbrowser
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from constants import *
from plotly_plots import plot_time_slider, plot_acoustic_spectra, plot_temperature_humidity, plot_parallel_selector
from mpl_plots import plot_similarity
from preprocessing import download_csv_if_needed, load_dataset

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
    
    start = HELSINKI_4DAYS_AGO
    end = HELSINKI_NOW
    sensors = [20, 21, 46, 109]
  
    # Code section bellow will check if required HTML pieces very already created.
    # It will skip a lot of time on the seconds run because it does not try to recreate existing HTML pieces.
    if all([
        os.path.exists(TIME_SLIDER_HTML),
        os.path.exists(ACOUSTIC_SPECTRA_HTML),
        os.path.exists(TEMPERATURE_HUMIDIY_HTML),
        os.path.exists(PARALLEL_SELECTOR_HTML),
    ]):
        with open(ACOUSTIC_SPECTRA_HTML, "r") as f:
            acoustic_spectra_html = f.read()
        with open(TEMPERATURE_HUMIDIY_HTML, "r") as f:
            temperature_humidity_html = f.read()
        with open(TIME_SLIDER_HTML, "r") as f:
            time_slider_html = f.read()
        with open(PARALLEL_SELECTOR_HTML, "r") as f:
            parallel_selector_html = f.read()
    else:
        csv_files = download_csv_if_needed(
                sensors,
                start.astimezone(UTC_TZ),
                end.astimezone(UTC_TZ),
                DATA_DIR
        )
        dataset = load_dataset(csv_files)

        acoustic_spectra_html = plot_acoustic_spectra(dataset, start, end)
        temperature_humidity_html = plot_temperature_humidity(dataset)
        time_slider_html = plot_time_slider(dataset)
        parallel_selector_html = plot_parallel_selector(dataset)
        _ = plot_similarity(dataset, start, end, OUTPUT_DIR)

    with open(ACOUSTIC_SPECTRA_INFO, "r") as f:
        acoustic_spectra_info = f.read()
    with open(TEMPERATURE_HUMIDIY_INFO, "r") as f:
        temperature_humidity_info = f.read()
    with open(SIMILARITY_INFO, "r") as f:
        similarity_info = f.read()

    html_data = {
        "time_slider_plot": time_slider_html,
        "acoustic_spectra_plot" : acoustic_spectra_html,
        "acoustic_spectra_info" : acoustic_spectra_info,
        "temperature_humidity_plot": temperature_humidity_html,
        "temperature_humidity_info": temperature_humidity_info,
        "parallel_selector_plot": parallel_selector_html,
        "similarity_info": similarity_info,
        "sensors": sensors,
        "OUTPUT_DIR": OUTPUT_DIR,
    }

    html_path = create_html(html_data)
    webbrowser.open(Path(html_path).absolute().as_uri())
