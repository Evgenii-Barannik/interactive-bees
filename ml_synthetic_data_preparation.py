import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import logging
from preprocessing import show_image

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

ML_DATA_DIR = "ml_data"
ML_SYNTHETIC_DATASET_NAME = "ml_synthetic_dataset.npz"

def gaussian(x, center, fwhm, amplitude):
    sigma = fwhm / 2.355
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def generate_synthetic_data_g1(time_array):
    channels = np.arange(2028)
    temperatures = 24 + 8 * np.cos(2 * np.pi * time_array - np.pi) 
    center_positions = (0.3 * 2028) + (0.5 * 2028 * np.cos(temperatures / 10 + np.pi))
    width_values = 120 + 60 * (np.sin(temperatures / 15))
    noise = np.random.rand(2028)*0.09
    position_and_width_values = zip(center_positions, width_values)
    spectra_array = np.array([
        0.1 + noise
        + gaussian(channels, position, width, 0.8)
        for (position, width) in position_and_width_values
        ])
    return time_array, temperatures, spectra_array

def generate_synthetic_data_g2(time_array):
    channels = np.arange(2028)
    temperatures = 24 + 8 * np.cos(2 * np.pi * time_array - np.pi) 
    g0_center_positions = (0.3 * 2028) + (0.5 * 2028 * np.cos(temperatures / 10 + np.pi))
    g0_width_values = np.full(shape=len(temperatures), fill_value=120)
    g1_center_positions = 0.5 * 2028 
    g1_width_values = 200 + 150 * (np.sin(temperatures / 10))
    noise = np.random.rand(2028)*0.09
    position_and_width_values = zip(g0_center_positions, g0_width_values, g1_width_values)
    spectra_array = np.array([
        0.1 + noise + 
        gaussian(channels, position, width, 0.4) +
        gaussian(channels, g1_center_positions, width_static, 0.4)
        for (position, width, width_static) in position_and_width_values
    ])
    return time_array, temperatures, spectra_array

def generate_synthetic_data_g3(time_array):
    channels = np.arange(2028)
    temperatures = 24 + 8 * np.cos(2 * np.pi * time_array - np.pi) 
    g0_center_positions = (0.5 * 2028) + (0.3 * 2028 * np.cos(temperatures / 10))
    g0_width_values = np.full(shape=len(temperatures), fill_value=120)
    g1_center_positions = 0.5 * 2028 
    g1_width_values = 200 + 150 * (np.sin(temperatures / 15))
    g2_center_positions = (0.5 * 2028) + (0.3 * 2028 * np.cos(temperatures / 5))
    g2_width_values = np.full(shape=len(temperatures), fill_value=300)
    noise = np.random.rand(2028)*0.09
    position_and_width_values = zip(g0_center_positions, g0_width_values, g1_width_values, g2_center_positions, g2_width_values)
    spectra_array = np.array([
        0.1 + noise + 
        gaussian(channels, p0, w0, 0.3) +
        gaussian(channels, g1_center_positions, w1, 0.3) +
        gaussian(channels, p2, w2, 0.3)
        for (p0, w0, w1, p2, w2) in position_and_width_values
    ])
    return time_array, temperatures, spectra_array

def create_dual_panel_animation(time_array, temperatures, spectra, pathname):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(time_array * 24, temperatures, 'b-', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature vs Time of Day')
    ax1.grid(True)
    ax1.set_ylim(15, 35)
    ax1.legend()
    ax2.set_xlim(0, 2028)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Synthetic spectral data')
    ax2.grid(True)
    scatter = ax2.scatter([], [], s=1, color='red')
    temp_point = ax1.scatter([], [], s=100, color='red', zorder=5)
    time_text = ax1.text(2, 33, '', fontsize=12, color='black')
    temp_text = ax2.text(100, 0.9, '', fontsize=12, color='black')
    
    def update(frame):
        current_time = time_array[frame] * 24
        current_temp = temperatures[frame]
        temp_point.set_offsets([[current_time, current_temp]])
        channels = np.arange(2028)
        scatter.set_offsets(np.column_stack([channels, spectra[frame]]))
        hours = int(current_time)
        minutes = int((current_time - hours) * 60)
        time_text.set_text(f'Time: {hours:02d} h\nTemperature: {current_temp:.1f}°C')
        temp_text.set_text(f'Time: {hours:02d} h\nTemperature: {current_temp:.1f}°C')
        return scatter, temp_point, time_text, temp_text
    
    anim = FuncAnimation(fig, update, frames=len(time_array), interval=100, blit=True)
    os.makedirs('plots', exist_ok=True)
    anim.save(pathname, writer='pillow')
    plt.close()

if __name__ == "__main__":
    normalized_time_array = np.linspace(0, 3, 72)
    time_array1, temperatures1, spectra1 = generate_synthetic_data_g1(normalized_time_array)
    time_array2, temperatures2, spectra2 = generate_synthetic_data_g2(normalized_time_array)
    time_array3, temperatures3, spectra3 = generate_synthetic_data_g3(normalized_time_array)
    
    # create_dual_panel_animation(time_array1, temperatures1, spectra1, 'plots/synthetic_data_g1_dual.gif')
    # show_image('plots/synthetic_data_g1_dual.gif')
    create_dual_panel_animation(time_array2, temperatures2, spectra2, 'plots/synthetic_data_g2_dual.gif')
    show_image('plots/synthetic_data_g2_dual.gif')
    # create_dual_panel_animation(time_array3, temperatures3, spectra3, 'plots/synthetic_data_g3_dual.gif')
    # show_image('plots/synthetic_data_g3_dual.gif')
    
