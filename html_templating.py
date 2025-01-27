from jinja2 import Environment, FileSystemLoader

from plotly_plots import * 
from mpl_plots import * 
from constants import *

def render_html(html_data):
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

    helsinki_tz = ZoneInfo('Europe/Helsinki')
    helsinki_now = datetime.now(helsinki_tz)
    helsinki_24h_ago = helsinki_now - pd.Timedelta(days=1)
    helsinki_days_ago = helsinki_now - pd.Timedelta(days=4)

    data_dir = "data"
    plots_dir = "plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    sensors = [20, 21, 46, 109]
    csv_files = download_csv_if_needed(sensors, helsinki_days_ago, helsinki_now, data_dir) 
    dataset = load_dataset(csv_files)

    # if os.path.exists("html_block.html"):
    #     with open("html_block.html", "r", encoding="utf-8") as f:
    #         html_block = f.read()
    # else:
    #     html_block = create_html_block()

    # with open(html_path, 'w') as file:
    #     averaged_plot_html, _ = create_averaged_plot_html(dataset, helsinki_days_ago, helsinki_now)
    #     phase_plot_html, _ = create_temp_humidity_plot_html(dataset, dataset["sensor"].values, helsinki_24h_ago, helsinki_now)
    #     file.write(averaged_plot_html + phase_plot_html)

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
