import pandas as pd
import numpy as np
import os
import requests
import pandas as pd
import logging
import math

from constants import *

def get_info_total(filtered_ds):
    num_of_datapoints = len(filtered_ds['datetime'].values)
    start_time = min(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    end_time = max(filtered_ds['datetime'].values).astimezone(HELSINKI_TZ)
    text_for_legend = f"First datapoint: {start_time}\nLast datapoint: {end_time}\nNumber of datapoints: {num_of_datapoints}\n" 
    return text_for_legend

def get_info_for_each_sensor(ds, start, end):
    text = ""
    all_sensors = np.unique(ds.sensor)
    for sensor_id in all_sensors:
        filtered_dataset = ds.where(
                (ds.sensor == sensor_id) &
                (ds.datetime >= start) & 
                (ds.datetime <= end),
                drop=True
        )
        if len(filtered_dataset.datetime.values) > 0:
            num_of_datapoints = len(filtered_dataset.datetime.values)
            start_time = min(filtered_dataset.datetime.values).astimezone(HELSINKI_TZ)
            end_time = max(filtered_dataset.datetime.values).astimezone(HELSINKI_TZ)
            transient_text = f"""\nFor sensor {sensor_id}:
First datapoint: {start_time}
Last datapoint: {end_time}
Number of datapoints: {num_of_datapoints}
""" 
            text += transient_text
    return text

# My CICD for images
def show_image(image_name):
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {image_name}')

# Database stores datetimes as UTC, so UTC datetimes are required for GET request
def download_csv_if_needed(sensors, start_datetime_utc, end_datetime_utc, dir_to_save_csv):
    assert start_datetime_utc.tzinfo == UTC_TZ
    assert end_datetime_utc.tzinfo == UTC_TZ

    assert start_datetime_utc <= end_datetime_utc
    date_start = start_datetime_utc.strftime('%Y-%m-%d') 
    date_end = end_datetime_utc.strftime('%Y-%m-%d') 
    downloaded_csvs = []
    os.makedirs(dir_to_save_csv, exist_ok=True) 
    base_url = "http://apiologia.zymologia.fi/export/"
    for sensor in sensors:
        pathname = f"{dir_to_save_csv}/sensor_{sensor}_from_{date_start}_to_{date_end}.csv"
        if not os.path.isfile(pathname):
            url = f"{base_url}?sensor={sensor}&date_from={date_start}&date_to={date_end}"
            logging.info(f"Using this request:\n{url}")
            response = requests.get(url)
            if response.status_code == 200:
                with open(pathname, 'wb') as file:
                    file.write(response.content)
                downloaded_csvs.append(pathname)           
                logging.info(f"CSV file downloaded and saved: {pathname}")
            else:
                logging.info(f"Failed to download CSV file for sensor {sensor}.\n{response}")
        else:
            downloaded_csvs.append(pathname)           
            logging.info(f"CSV file already exists: {pathname}")
    return downloaded_csvs

def parse_spectrum(spectrum_string):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    spectrum = [float(s) for s in list_of_strings]
    return spectrum

def load_dataset(files_to_load, filter_by_datetime = False, start = None, end = None): 
    assert files_to_load
    populated_rows = []
    for pathname in files_to_load:
        df_with_raw_spectrum = pd.read_csv(pathname) 
        for _, row in df_with_raw_spectrum.iterrows():
            row_data = row.to_dict()
            row_data["DateTime"] = pd.to_datetime(row_data["DateTime"]) # Datetime strings are converted to Pandas DateTimes
            row_data["Spectrum"] = parse_spectrum(row_data["Spectrum"])
            populated_rows.append(row_data)

    transient_dataframe = pd.DataFrame(populated_rows)
    transient_dataframe = transient_dataframe.rename(columns= lambda x: str(x).lower())
    dataset = transient_dataframe.to_xarray()

    all_sensors = np.unique(dataset.sensor)
    for sensor_id in all_sensors:
        filtered_dataset = dataset.where(dataset.sensor==sensor_id, drop=True)

        spectra_len_values = np.unique([len(s) for s in filtered_dataset.spectrum.values])
        frequency_min_values = np.unique([v for v in filtered_dataset.frequency_min])
        frequency_max_values = np.unique([v for v in filtered_dataset.frequency_max])
        frequency_start_index_values = np.unique([v for v in filtered_dataset.frequency_start_index.values])
        frequency_scaling_factor_values = np.unique([v for v in filtered_dataset.frequency_scaling_factor.values])

        assert len(spectra_len_values) == 1
        assert len(frequency_min_values) == 1
        assert len(frequency_max_values) == 1
        assert len(frequency_start_index_values) == 1
        assert len(frequency_scaling_factor_values) == 1

        spectra_len = spectra_len_values[0]
        frequency_min = frequency_min_values[0]
        frequency_max = frequency_max_values[0]
        frequency_start_index = frequency_start_index_values[0]
        frequency_scaling_factor = frequency_scaling_factor_values[0]
        frequencies = [
                (bin+frequency_start_index)*frequency_scaling_factor
                for bin in range(spectra_len) 
        ]

        assert spectra_len == len(frequencies)
        assert math.isclose(frequency_min, frequencies[0])
        assert math.isclose(frequency_max, frequencies[-1])

        logging.info("Spectra for sensor {:3}: ({:.2f}-{:-7.2f} Hz) ({} bins)".format(
            sensor_id, frequency_min, frequency_max, spectra_len, spectra_len
        ))

    if filter_by_datetime:
        assert start
        assert end
        assert start < end
        dataset = dataset.where(
            (dataset.datetime >= start) &
            (dataset.datetime <= end),
            drop=True,
            other = 0
        )
    # Usa of xarray.DataArray.where() without specification of non-nan other
    # ends up in converting columns to float

    logging.info(f"\n{dataset}")
    logging.info("Memory used for dataset: {:.3f} MB".format(dataset.nbytes / (1024**2)))
    logging.info("Type used for sensor: {}".format(type(dataset["sensor"].values[0])))
    logging.info("Type used for datetime: {}".format(type(dataset["datetime"].values[0])))
    logging.info("First DateTime in dataset (UTC): {}".format(min(dataset["datetime"].values).astimezone(UTC_TZ)))
    logging.info("Last DateTime in dataset (UTC):  {}".format(max(dataset["datetime"].values).astimezone(UTC_TZ)))
    logging.info("First DateTime in dataset (Helsinki): {}".format(min(dataset["datetime"].values).astimezone(HELSINKI_TZ)))
    logging.info("Last DateTime in dataset (Helsinki):  {}".format(max(dataset["datetime"].values).astimezone(HELSINKI_TZ)))

    return dataset

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
    sensors = [109, 116, 117, 118]
    files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
    )

    filtered_dataset = load_dataset(files, True, start, end)
    logging.info(get_info_for_each_sensor(filtered_dataset, start, end))
