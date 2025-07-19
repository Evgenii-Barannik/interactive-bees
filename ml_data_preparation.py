import pandas as pd
import numpy as np
import os
import logging
from preprocessing import download_csv_if_needed, load_dataset
from constants import *

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)

ML_DATA_DIR = "ml_data"
ML_DATASET_NAME = "ml_dataset.npz"

def prepare_ml_dataset():
    ml_dataset = os.path.join(ML_DATA_DIR, ML_DATASET_NAME)
    if os.path.exists(ml_dataset):
        logging.info("Dataset already exists, loading it.")
        data = np.load(ml_dataset)
        df = pd.DataFrame({
            'datetime': data['datetime'],
            'sensor': data['sensor'],
            'temperature': data['temperature'],
            'spectrum': list(data['spectrum'])
        })
        return df

    logging.info("Dataset not found, preparing it.")
    sensors = [109]
    start = datetime(2025, 7, 12, 12).astimezone(UTC_TZ)
    end = datetime(2025, 7, 12, 12).astimezone(UTC_TZ) + pd.Timedelta(days=7)
    
    csv_files = download_csv_if_needed(
        sensors,
        start,
        end,
        DATA_DIR
    )
    dataset = load_dataset(csv_files, True, start, end)
    ml_data = []
    for i in range(len(dataset.datetime)):
        ml_data.append({
            'datetime': dataset.datetime.values[i],
            'sensor': dataset.sensor.values[i],
            'temperature': dataset.temperature.values[i],
            'spectrum': dataset.spectrum.values[i]
        })
    
    ml_df = pd.DataFrame(ml_data)
    # Corrupted spectra removal
    skip_frames_1based = [1, 4, 15, 19, 24, 27, 29, 42, 46, 80, 81, 82, 150, 155]
    skip_idx = [i-1 for i in skip_frames_1based]
    if skip_idx:
        ml_df = ml_df.drop(index=skip_idx).reset_index(drop=True)
    
    os.makedirs(ML_DATA_DIR, exist_ok=True)
    np.savez(
        ml_dataset,
        datetime=ml_df['datetime'].values,
        sensor=ml_df['sensor'].values,
        temperature=ml_df['temperature'].values,
        spectrum=np.array(ml_df['spectrum'].values.tolist())
    )
    return ml_df

if __name__ == "__main__":
    ml_data = prepare_ml_dataset()
    print(f"Data shape: {ml_data.shape}")
    print(f"Sensors: {ml_data['sensor'].unique()}")
    print(f"Temperature range: {ml_data['temperature'].min():.2f} - {ml_data['temperature'].max():.2f} °C") 
