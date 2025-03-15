import numpy as np
from lmfit import Model, Parameters
import logging
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from constants import *
from plotly_plots import normalize_spectrum
from preprocessing import get_info_total, load_dataset, show_image, download_csv_if_needed

FITTING_MODEL = [
        {
            'type': 'background',
            'slope_guess': 0,
            'intercept_guess': 10,
        },
        {
            'type': 'peak',
            'center_range': (70, 155),
            'amplitude_guess': 70,
            'fwhm_guess': 30,
        },
        {  
            'type': 'peak',
            'center_range': (175, 225),
            'amplitude_guess': 40,
            'fwhm_guess': 60,
        },
        {  
            'type': 'peak',
            'center_range': (225, 275),
            'amplitude_guess': 40,
            'fwhm_guess': 60,
        },
        {  
            'type': 'peak',
            'center_range': (300, 350),
            'amplitude_guess': 40,
            'fwhm_guess': 50,
            'fwhm_max': 70
        },
        {  
            'type': 'peak',
            'center_range': (370, 440),
            'amplitude_guess': 20,
            'fwhm_guess': 30,
            'fwhm_max': 70
        },
        {  
            'type': 'peak',
            'center_range': (450, 500),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (500, 520),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        },
        {  
            'type': 'peak',
            'center_range': (550, 650),
            'amplitude_guess': 10,
            'fwhm_guess': 30
        }
    ]

def linear_background(x, slope, intercept):
    return slope * x + intercept

def gaussian(x, amplitude, center, fwhm):
    sigma = fwhm / 2.355
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def create_model():
    model = Model(linear_background, prefix='bg_')
    for i in range(1, len(FITTING_MODEL)):
        model += Model(gaussian, prefix=f'g{i}_')
    return model

def plot_gaussians(ds, start, end, output_path, name_overide=None):
    images = []
    all_sensors = np.unique(ds.sensor)

    spectral_colormap = mpl.colormaps['turbo']

    for sensor_id in all_sensors:
        filtered_ds = ds.where (
            (ds.sensor == sensor_id) &
            (ds['datetime'] > start) & 
            (ds['datetime'] < end),
            drop = True,
            other = 0
        )
        if len(filtered_ds['datetime'].values) == 0: # There must be some datapoints to plot a graph
            continue
        raw_spectra = np.vstack(filtered_ds['spectrum'].values)
        averaged_spectrum = normalize_spectrum(np.nanmean(raw_spectra, axis=0))
        spectrum_len = len(averaged_spectrum)

        freq_factor = filtered_ds['frequency_scaling_factor'].values[0]
        freq_start  = filtered_ds['frequency_start_index'].values[0]
        frequencies = np.array([(bin+freq_start)*freq_factor for bin in range(spectrum_len)])
                                               
        x_full = frequencies
        y_full = averaged_spectrum
        
        window_min = 60
        window_max = 650
        mask = (x_full >= window_min) & (x_full <= window_max)
        x_masked = x_full[mask]
        y_masked = y_full[mask]
        
        model = create_model()
        params = Parameters()
        bg = FITTING_MODEL[0]
        params.add('bg_slope', value=bg['slope_guess'], vary=False)
        params.add('bg_intercept', value=bg['intercept_guess'], min=0, max=np.min(y_masked))
        for i, cfg in enumerate(FITTING_MODEL):
            if cfg['type'] == 'peak':
                prefix = f'g{i}_'
                params.add(f'{prefix}amplitude', value=cfg['amplitude_guess'], min=0)
                params.add(f'{prefix}center', value=np.mean(cfg['center_range']), min=cfg['center_range'][0], max=cfg['center_range'][1])
                params.add(f'{prefix}fwhm', value=cfg['fwhm_guess'], min=cfg.get('fwhm_min', 10), max=cfg.get('fwhm_max', 100))
        result = model.fit(y_masked, params, x=x_masked)
        components = result.eval_components(x=x_masked)
        
        # Calculate RMSE only
        residuals = y_masked - result.best_fit
        rmse = np.sqrt(np.mean(residuals**2))
        logging.info(f"RMSE for sensor {sensor_id}: {rmse:.4f}")
        
        fig, axes = plt.subplots(
                2, 1,
                figsize=(14, 8),
                gridspec_kw={'height_ratios': [5, 1]},
                sharex=True
        )
        ax1 = axes[0]
        ax2 = axes[1]
        ax1.grid(visible=True)
        ax2.grid(visible=True)
        loc = plticker.MultipleLocator(base=100.0) # this locator puts ticks at regular intervals

        ax1.xaxis.set_major_locator(loc)

        ax1.set_ylabel('Amplitude, %', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
        
        ax2.set_xlabel('Frequency, Hz', fontsize=14)
        ax2.set_ylabel('Residuals, %', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12, length=6, width=1.5)
        
        ax1.plot(x_full, y_full, 'b-', label='Acoustic spectrum')
        ax1.plot(x_masked, result.best_fit, 'k--', label='Total model')
        ax1.plot(x_masked, components['bg_'], linestyle='dotted', label='Linear background')

        # Legend patching
        p = result.params
        gauss_count = sum(1 for name in components.keys() if name.startswith('g'))
        for i, (name, comp) in enumerate(components.items(), 1):
            if name.startswith('g'):
                j = i - 1
                amplitude = p[f'g{j}_amplitude'].value
                center = p[f'g{j}_center'].value
                fwhm = p[f'g{j}_fwhm'].value
                color = spectral_colormap(j / gauss_count)
                
                ax1.plot(
                    x_masked,
                    comp,
                    color=color,
                    label=f"Gauss peak {j}: {center:>6.1f} Hz, {fwhm:>6.1f} Hz, {amplitude:>6.1f}"
                )
        handles, _ = ax1.get_legend_handles_labels()
        patch0 = mpatches.Patch(color='None', label=f"Gauss peak N: Center, FWHM, Amplitude")
        handles.append(patch0) 

        metrics_text = f"\nModel RMSE: {rmse:.4f}"
        patch_metrics = mpatches.Patch(color='None', label=metrics_text)
        handles.append(patch_metrics)

        window_text = f"Window used for fitting: {window_min} to {window_max} Hz"
        patch_metrics = mpatches.Patch(color='None', label=window_text)
        handles.append(patch_metrics)

        datapoints_info = "\nFit for normalized averaged acoustic spectrum\n{}Sensor: {}\n".format(get_info_total(filtered_ds), sensor_id, window_min, window_max)
        patch1 = mpatches.Patch(color='None', label=datapoints_info) 
        handles.append(patch1) 
        fig.legend(bbox_to_anchor=(0.98, 0.98), handles=handles, fontsize=9)

        ax1.set_ylim(0, 100)
        ax1.set_xlim(0, 700)
        residuals = y_masked - result.best_fit
        ax2.plot(x_masked, residuals, 'k-', label='Residuals')
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)

        plt.tight_layout()
        # plt.subplots_adjust(right=0.75)

        if name_overide:
            img_pathname = os.path.join(output_path, name_overide)
        else:
            img_pathname = os.path.join(output_path, f"gaussians-sensor-{sensor_id}.png")

        images.append(img_pathname) 
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(img_pathname, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"PNG file {img_pathname} was created!")
    return images

def plot_gauss_example():
    sensors = [116]
    start = datetime(2025, 2, 13, 0, tzinfo = HELSINKI_TZ)
    end = datetime(2025, 2, 17, 0, 0, tzinfo = HELSINKI_TZ)
    csv_files = download_csv_if_needed(
            sensors,
            start.astimezone(UTC_TZ),
            end.astimezone(UTC_TZ),
            DATA_DIR
            )
    filtered_ds = load_dataset(csv_files, True, start, end)
    gauss_plots = plot_gaussians(filtered_ds, start, end, OUTPUT_DIR, "gauss_example.png")
    return gauss_plots

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler()  # Log to console
        ]
    )

    gauss_example = plot_gauss_example()
    show_image(gauss_example[0])
