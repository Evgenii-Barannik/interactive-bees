import pandas as pd
import numpy as np
import xarray as xr
import os
import requests
import pandas as pd
import logging
import math

from constants import * 

def get_non_nan_spectrum(spectrum):
    spectrum = np.asarray(spectrum)
    assert np.any(~np.isnan(spectrum)), "Array contains only NaN values."
    return spectrum[~np.isnan(spectrum)]

def get_info_about_filtered_datapoints(filtered_dataset):
    num_of_datapoints = filtered_dataset['datetime'].shape[0]
    start_time = min(filtered_dataset['datetime'].values).astimezone(HELSINKI_TZ)
    end_time = max(filtered_dataset['datetime'].values).astimezone(HELSINKI_TZ)
    text_for_legend = f"First datapoint: {start_time}\nLast datapoint: {end_time}\nNumber of datapoints: {num_of_datapoints}\n" 
    return text_for_legend

def get_info_about_all_datapoints(ds, start, end):
    text = ""
    for sensor_id in ds["sensor"].values:
        filtered_dataset = ds.sel(sensor=sensor_id).where(
                (ds['datetime'] >= start) & 
                (ds['datetime'] <= end) & 
                ds.sel(sensor=sensor_id)['base'].notnull(),
                drop=True
        )
        if filtered_dataset['datetime'].shape[0] > 0:
            num_of_datapoints = filtered_dataset['datetime'].shape[0]
            start_time = min(filtered_dataset['datetime'].values).astimezone(HELSINKI_TZ)
            end_time = max(filtered_dataset['datetime'].values).astimezone(HELSINKI_TZ)
            transient_text = f"""For sensor {sensor_id}:
First datapoint: {start_time}
Last datapoint: {end_time}
Number of datapoints: {num_of_datapoints}\n
""" 
            text += transient_text
    return text

# My CICD for images
def show_image(image_name):
    lyn_app_path = "/Applications/Lyn.app"
    if os.path.exists(lyn_app_path):
        os.system(f'open -g -a {lyn_app_path} {image_name}')

def download_csv_if_needed(sensors, start_datetime_utc, end_datetime_utc, dir_to_save_csv):
    # Database stores datetimes as UTC, so UTC datetimes are required for GET request
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
            response = requests.get(url)
            if response.status_code == 200:
                with open(pathname, 'wb') as file:
                    file.write(response.content)
                downloaded_csvs.append(pathname)           
                logging.info(f"CSV file downloaded and saved: {pathname}")
            else:
                logging.info(f"Failed to download CSV file for sensor {sensor}. Status code: {response.status_code}")
        else:
            downloaded_csvs.append(pathname)           
            logging.info(f"CSV file already exists: {pathname}")
    return downloaded_csvs

def get_max_spectrum_len(files_to_load):
    max_spectrum_len = 0
    for pathname in files_to_load:
        df_with_raw_spectra = pd.read_csv(pathname)
        for _, row in df_with_raw_spectra.iterrows():
            list_of_strings = str(row['Spectrum']).strip('[]').split(';')
            spectrum_len = len([float(s) for s in list_of_strings])
            if spectrum_len > max_spectrum_len:
                max_spectrum_len = spectrum_len

    assert max_spectrum_len != 0, f"No data found inside passed CSV files: {files_to_load}"
    return max_spectrum_len

def parse_and_pad_spectrum(spectrum_string, required_spectrum_length):
    list_of_strings = str(spectrum_string).strip('[]').split(';')
    spectrum = [float(s) for s in list_of_strings]
    assert len(spectrum) <= required_spectrum_length
    if len(spectrum) < required_spectrum_length:
        spectrum.extend ([np.nan]*(required_spectrum_length - len(spectrum))) # Spectrum array get extended at the end
    assert len(spectrum) == required_spectrum_length 
    return spectrum

def load_dataset(files_to_load, filter = False, start = None, end = None):
    assert files_to_load
    max_spectrum_len = get_max_spectrum_len(files_to_load)
    populated_rows = []
    for pathname in files_to_load:
        df_with_raw_spectrum = pd.read_csv(pathname) 
        for _, row in df_with_raw_spectrum.iterrows():
            row_data = row.to_dict()
            row_data["DateTime"] = pd.to_datetime(row_data["DateTime"]) # Datetime strings are converted to Pandas DateTimes
            row_data["Spectrum"] = parse_and_pad_spectrum(row_data["Spectrum"], max_spectrum_len)
            populated_rows.append(row_data)

    # Getting possible coord values
    transient_dataframe = pd.DataFrame(populated_rows)
    unique_sensors = transient_dataframe['Sensor'].unique()
    unique_datetimes = transient_dataframe['DateTime'].unique()

    # Initializing arrays with NaNs
    columns = transient_dataframe.keys()
    spectrum_array = np.full((len(unique_datetimes), len(unique_sensors), max_spectrum_len), np.nan) #3D array
    other_arrays = {
            c.lower(): np.full((len(unique_datetimes), len(unique_sensors)), np.nan) #2D arrays
            for c in columns if c not in ['DateTime', 'Sensor', 'Spectrum']
    }

    # Populating arrays 
    for _, row in transient_dataframe.iterrows():
        datetime_idx = list(unique_datetimes).index(row['DateTime'])
        sensor_idx = list(unique_sensors).index(row['Sensor'])
        spectrum_array[datetime_idx, sensor_idx, :len(range(max_spectrum_len))] = row['Spectrum']
        for c in columns:
            if c not in ['DateTime', 'Sensor', 'Spectrum']:
                other_arrays[c.lower()][datetime_idx, sensor_idx] = row[c]

    # Creating dataset
    dataset = xr.Dataset(
        { "spectrum": (['datetime', 'sensor', 'channel'], spectrum_array) } |
        { key: (['datetime', 'sensor'], value) for key, value in other_arrays.items() },
        coords={
            "datetime": unique_datetimes,
            "sensor": unique_sensors,
            "channel": np.arange(max_spectrum_len)
        }
    )
    
    logging.info("\nDataset:\n{}".format(dataset))
    all_sensors = dataset["sensor"].values
    for sensor_id in all_sensors:
        filtered_dataset = dataset.sel(sensor=sensor_id).where(
                dataset.sel(sensor=sensor_id)['base'].notnull(),
                drop=True
        )
        # TODO: Check if all values are the same in rows
        spectra_len = filtered_dataset['spectrum'][0].shape[0]
        spectra_len_non_nan = get_non_nan_spectrum(filtered_dataset['spectrum'][0]).shape[0]
        frequency_min = filtered_dataset['frequency_min'].values[0]
        frequency_max = filtered_dataset['frequency_max'].values[0]
        frequency_start_index = filtered_dataset['frequency_start_index'].values[0]
        frequency_scaling_factor = filtered_dataset['frequency_scaling_factor'].values[0]
        frequencies = [
                    (bin+frequency_start_index)*frequency_scaling_factor
                    for bin in range(spectra_len_non_nan) 
        ]
        assert spectra_len_non_nan == len(frequencies)
        assert spectra_len_non_nan == filtered_dataset['spectrum_length'].values[0]
        assert math.isclose(frequency_min, frequencies[0])
        assert math.isclose(frequency_max, frequencies[-1])
        logging.info("Spectra for sensor {:3}: ({:.2f}-{:-7.2f} Hz) ({} bins) ({} bins before NaN stripping) ".format(
            sensor_id, frequency_min, frequency_max, spectra_len_non_nan, spectra_len
        ))

    if filter:
        assert start
        assert end
        assert start < end
        dataset = dataset.where(
            (dataset['datetime'] >= start) &
            (dataset['datetime'] <= end),
            drop=True
        )
    
    logging.info("Type used for DateTime: {}".format(type(dataset["datetime"].values[0])))
    logging.info("Memory used for dataset: {:.3f} MB".format(dataset.nbytes / (1024**2)))
    logging.info("First DateTime in dataset (UTC TZ): {}".format(min(dataset["datetime"].values).astimezone(UTC_TZ)))
    logging.info("Last DateTime in dataset (UTC TZ):  {}".format(max(dataset["datetime"].values).astimezone(UTC_TZ)))
    logging.info("First DateTime in dataset (Helsinki TZ): {}".format(min(dataset["datetime"].values).astimezone(HELSINKI_TZ)))
    logging.info("Last DateTime in dataset (Helsinki TZ):  {}\n".format(max(dataset["datetime"].values).astimezone(HELSINKI_TZ)))
    return dataset

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    sensors = [20, 21, 46, 109]
    files = download_csv_if_needed(
            sensors,
            HELSINKI_24HOURS_AGO.astimezone(UTC_TZ),
            HELSINKI_NOW.astimezone(UTC_TZ),
            DATA_DIR
    )
    filtered_dataset = load_dataset(files, True, HELSINKI_24HOURS_AGO, HELSINKI_NOW)
    logging.info(get_info_about_all_datapoints(filtered_dataset, HELSINKI_4DAYS_AGO, HELSINKI_NOW))
