from jinja2 import Environment, FileSystemLoader

from plotly_plots import * 
from mpl_plots import * 
from constants import *

def create_html(html_data):
    env = Environment(loader=FileSystemLoader('templates'))
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

    sensors = [20, 21, 46, 109]
  
    # Code section bellow will check if required HTML pieces very already created.
    # It will skip a lot of time on the seconds run because it does not try to recreate existing HTML pieces.
    if all([
        os.path.exists(ACOUSTIC_SPECTRA_PLOT_PATHNAME),
        os.path.exists(ACOUSTIC_SPECTRA_INFO_PATHNAME),
        os.path.exists(TEMPERATURE_HUMIDIY_PLOT_PATHNAME),
        os.path.exists(TEMPERATURE_HUMIDIY_INFO_PATHNAME),
        os.path.exists(SIMILARITY_INFO_PATHNAME),
    ]):
        with open(ACOUSTIC_SPECTRA_PLOT_PATHNAME, "r") as f:
            acoustic_spectra_plot = f.read()
        with open(ACOUSTIC_SPECTRA_INFO_PATHNAME, "r") as f:
            acoustic_spectra_info = f.read()
        with open(TEMPERATURE_HUMIDIY_PLOT_PATHNAME, "r") as f:
            temperature_humidity_plot = f.read()
        with open(TEMPERATURE_HUMIDIY_INFO_PATHNAME, "r") as f:
            temperature_humidity_info = f.read()
        with open(SIMILARITY_INFO_PATHNAME, "r") as f:
            similarity_info = f.read()
    else:
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

    # Part below only test templating with already created HTML pieces.
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
