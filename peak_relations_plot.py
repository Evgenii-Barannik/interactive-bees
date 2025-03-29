import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import time

from datetime import datetime
from constants import *
from plotly_plots import normalize_spectrum
from preprocessing import get_info_total, load_dataset, show_image, download_csv_if_needed
from similarity_plot import get_ticks_for_helsinki_tz, format_time_to_helsinki
from gauss_plot import fit_model

MAX_RATIO_VALUE = 20

def plot_peak_relations(ds, sensors_to_draw, start, end, output_path):
    start_time = time.time()
    logging.info(f"Plotting peak relations for requested range\nSTART:   {start}\nEND:     {end}")
    relation_types = [(f"Peak {i}/Peak 1", i, 1) for i in range(2, 9)]
    images = []
    
    for sensor_id in sensors_to_draw:
        filtered_ds = ds.where(
            (ds.sensor == sensor_id) &
            (ds['datetime'] >= start) & 
            (ds['datetime'] <= end),
            drop=True,
            other=0
        )
        
        if len(filtered_ds['datetime'].values) == 0:  # There must be some datapoints to plot a graph
            continue
        
        measurement_datetimes = np.array([dt.astimezone(HELSINKI_TZ) for dt in filtered_ds['datetime'].values])
        spectra = np.vstack(filtered_ds['spectrum'].values)
        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start = filtered_ds['frequency_start_index'].values[0]
        spectra_len = spectra.shape[1]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectra_len)])
        
        relations_data = {
            'datetime': measurement_datetimes,
            'timestamp': [dt.timestamp() for dt in measurement_datetimes]
        }
        
        for j, spectrum in enumerate(spectra):
            normalized_spectrum = normalize_spectrum(spectrum, frequencies, NORMALIZATION_LIMIT)
            (_, result, _, _) = fit_model(frequencies, normalized_spectrum)
            p = result.params
            
            for i in range(1, len(FITTING_MODEL)):
                if FITTING_MODEL[i]['type'] == 'peak':
                    prefix = f'g{i}_'
                    amplitude = p[f'{prefix}amplitude'].value
                    fwhm = p[f'{prefix}fwhm'].value
                    intensity = amplitude * fwhm
                    
                    col_name = f'peak_{i}_intensity'
                    if col_name not in relations_data:
                        relations_data[col_name] = np.zeros(len(measurement_datetimes))
                    relations_data[col_name][j] = intensity
        
        df = pd.DataFrame(relations_data)
        for _, numerator_idx, denominator_idx in relation_types:
            num_col = f'peak_{numerator_idx}_intensity'
            den_col = f'peak_{denominator_idx}_intensity'
            
            if num_col in df.columns and den_col in df.columns:
                relation_col = f'relation_{numerator_idx}_{denominator_idx}'
                df[relation_col] = df[num_col] / df[den_col]
                
                df[relation_col] = df[relation_col].replace([np.inf, -np.inf], np.nan)
                df[relation_col] = df[relation_col].fillna(0)
                
                filtered_points = (df[relation_col] > MAX_RATIO_VALUE).sum()
                if filtered_points > 0:
                    logging.info(f"Filtered {filtered_points} points with ratio > {MAX_RATIO_VALUE} for {relation_col}")
                
                df[relation_col] = np.minimum(df[relation_col], MAX_RATIO_VALUE)
        
        gauss_count = sum(1 for cfg in FITTING_MODEL if cfg['type'] == 'peak')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.grid(True, alpha=0.3)
        
        for _, numerator_idx, denominator_idx in relation_types:
            relation_col = f'relation_{numerator_idx}_{denominator_idx}'
            if relation_col in df.columns:
                color = COLORMAP_FOR_GAUSSIANS(numerator_idx / gauss_count)
                label = f"Peak {numerator_idx}/Peak {denominator_idx}"
                ax.plot(df['timestamp'], df[relation_col], 'o-', color=color, label=label)
        
        datetimes_for_ticks = get_ticks_for_helsinki_tz(start, end)
        timestamps_for_ticks = [d.timestamp() for d in datetimes_for_ticks]
        ax.set_xticks(timestamps_for_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(format_time_to_helsinki))
        ax.set_xlabel('DateTime', fontsize=14)
        ax.set_ylabel('Peak Intensity Ratio', fontsize=14)
        ax.set_xlim(start.timestamp(), end.timestamp())
        
        handles = []
        for _, numerator_idx, denominator_idx in relation_types:
            relation_col = f'relation_{numerator_idx}_{denominator_idx}'
            if relation_col in df.columns:
                color = COLORMAP_FOR_GAUSSIANS(numerator_idx / gauss_count)
                handles.append(mpatches.Patch(color=color, label=f'Peak {numerator_idx}/Peak {denominator_idx}'))
        
        datapoints_info = "\nRelations between peak intensities\nfor fitted individual acoustic spectra\n{}Sensor: {}".format(
            get_info_total(filtered_ds), sensor_id
        )
        patch = mpatches.Patch(color='None', label=datapoints_info)
        handles.append(patch)
        window_text = f"\nWindow used for fitting: {FITTING_WINDOW_MIN} to {FITTING_WINDOW_MAX} Hz"
        patch = mpatches.Patch(color='None', label=window_text)
        handles.append(patch)
        fig.legend(bbox_to_anchor=(0.98, 0.98), handles=handles, fontsize=9)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.70)
        
        first_datetime = min(measurement_datetimes)
        last_datetime = max(measurement_datetimes)
        img_pathname = create_artifact_pathname(
            "peak_relations", output_path, sensor_id, first_datetime, last_datetime, "png"
        )
        
        images.append(img_pathname)
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    
    total_time = time.time() - start_time
    logging.info(f"Total plot creation time: {total_time:.2f} seconds")
    return images

def plot_peak_relations_example():
    sensors = [116]
    start = datetime(2025, 2, 13, 0, tzinfo=HELSINKI_TZ)
    end = datetime(2025, 2, 17, 0, 0, tzinfo=HELSINKI_TZ)
    
    csv_files = download_csv_if_needed(
        sensors,
        start.astimezone(UTC_TZ),
        end.astimezone(UTC_TZ),
        DATA_DIR
    )
    filtered_ds = load_dataset(csv_files, True, start, end)
    
    relation_plots = plot_peak_relations(
        filtered_ds,
        sensors,
        start,
        end,
        OUTPUT_DIR,
    )
    return relation_plots


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )
    
    relation_plots = plot_peak_relations_example()
    show_image(relation_plots[0]) 
