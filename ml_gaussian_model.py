import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import subprocess

from ml_synthetic_data_preparation import generate_synthetic_data_g1, generate_synthetic_data_g2, generate_synthetic_data_g3
from ml_data_preparation import prepare_ml_dataset
from preprocessing import show_image

def normalize_temperatures(temperatures):
    temp_mean = temperatures.mean()
    temp_std = temperatures.std()
    temps_norm = (temperatures - temp_mean) / temp_std
    return temps_norm, temp_mean, temp_std

NEURONS_IN_LAYER = 16

class GaussianNetwork(nn.Module):
    def __init__(self, frequencies, n_gaussians=1):
        super().__init__()
        self.n_gaussians = n_gaussians
        frequencies = frequencies.reshape(-1).to(dtype=torch.float32) # Flattens spectrum
        self.register_buffer("frequencies", frequencies)
        
        # Separate networks for each Gaussian
        self.A_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, NEURONS_IN_LAYER),
                nn.Tanh(),
                nn.Linear(NEURONS_IN_LAYER, 1)
            ) for _ in range(n_gaussians)
        ])
        
        self.mu_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, NEURONS_IN_LAYER),
                nn.Tanh(),
                nn.Linear(NEURONS_IN_LAYER, 1)
            ) for _ in range(n_gaussians)
        ])
        
        self.sigma_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, NEURONS_IN_LAYER),
                nn.Tanh(),
                nn.Linear(NEURONS_IN_LAYER, 1)
            ) for _ in range(n_gaussians)
        ])
        
        # Constant baseline
        self.baseline = nn.Parameter(torch.tensor(0.1))
        
        # Initialization with realistic values
        with torch.no_grad():
            freq_min = float(self.frequencies.min())
            freq_max = float(self.frequencies.max())
            freq_span = freq_max - freq_min
            centers = torch.linspace(freq_min + 0.2 * freq_span, freq_min + 0.4 * freq_span, n_gaussians)
            self.base_mu = torch.tensor((freq_min + freq_max) * 0.5)
            for g in range(n_gaussians):
                self.A_nets[g][-1].weight.fill_(0.0)
                self.A_nets[g][-1].bias.fill_(1.0)
                
                self.mu_nets[g][-1].weight.fill_(0.0)
                self.mu_nets[g][-1].bias.fill_(centers[g] - self.base_mu)
                
                self.sigma_nets[g][-1].weight.fill_(0.0)
                self.sigma_nets[g][-1].bias.fill_(4.8)

    def forward(self, temperature):
        batch_size = temperature.shape[0]
        f = self.frequencies.view(1, -1)
        
        A_list = []
        mu_list = []
        sigma_list = []
        
        for g in range(self.n_gaussians):
            A_g = torch.sigmoid(self.A_nets[g](temperature)) 
            mu_g = self.base_mu + self.mu_nets[g](temperature) 
            sigma_g = torch.exp(self.sigma_nets[g](temperature))
            
            A_list.append(A_g)
            mu_list.append(mu_g)
            sigma_list.append(sigma_g)
        
        A = torch.cat(A_list, dim=1)
        mu = torch.cat(mu_list, dim=1)
        sigma = torch.cat(sigma_list, dim=1)
        
        mu = mu.view(-1, self.n_gaussians, 1)
        sigma = sigma.view(-1, self.n_gaussians, 1)
        
        gaussians = A.view(-1, self.n_gaussians, 1) * torch.exp(-0.5 * ((f - mu) / sigma)**2)
        total_gaussian = gaussians.sum(dim=1)
        return self.baseline + total_gaussian

def plot_sythetic_fitting_results(temperatures_norm, temperatures_orig, spectra, model, losses):
    fig = plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.plot(losses)
    plt.title(f'Loss Function, final = {losses[-1]:.6f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    temp_range = np.linspace(temperatures_orig.min(), temperatures_orig.max(), 5)
    colors = ['red', 'orange', 'green', 'blue', 'violet']
    
    for i, target_temp in enumerate(temp_range):
        idx = torch.argmin(torch.abs(temperatures_orig.squeeze() - target_temp)).item()
        temp = temperatures_orig[idx].item()
        with torch.no_grad():
            y_pred = model(temperatures_norm[idx:idx+1])
            
            # Get individual Gaussians
            A_list, mu_list, sigma_list = [], [], []
            for g in range(model.n_gaussians):
                A_list.append(torch.sigmoid(model.A_nets[g](temperatures_norm[idx:idx+1])))
                mu_list.append(model.base_mu + model.mu_nets[g](temperatures_norm[idx:idx+1]))
                sigma_list.append(torch.exp(model.sigma_nets[g](temperatures_norm[idx:idx+1])))
            
            A = torch.cat(A_list, dim=1)
            mu = torch.cat(mu_list, dim=1)
            sigma = torch.cat(sigma_list, dim=1)
            
            f_plot = model.frequencies.view(1, -1)
            individual_gaussians = []
            for g in range(model.n_gaussians):
                gaussian = A[0, g] * torch.exp(-0.5 * ((f_plot - mu[0, g]) / sigma[0, g])**2)
                individual_gaussians.append(gaussian.squeeze())
        
        plt.subplot(2, 3, i+2)
        x_np = np.arange(spectra.shape[1])
        plt.plot(x_np, spectra[idx].numpy(), 'b-', alpha=0.7, label='True')
        plt.plot(x_np, y_pred.squeeze().numpy(), 'r-', label='Predicted')
        
        for g in range(model.n_gaussians):
            plt.plot(x_np, individual_gaussians[g].numpy(), '--', 
                    color=colors[g % len(colors)], alpha=0.6, 
                    label=f'Gaussian {g+1}')
        
        plt.title(f'T = {temp:.1f}°C')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/gaussian_model_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    subprocess.run(['open', plot_path])

def save_loss_curve(losses, path='plots/dataset_training_loss.png'):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title(f'Loss Function, final = {losses[-1]:.6f}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    subprocess.run(['open', path])

def train_model(
    temperatures,
    spectra,
    n_gaussians,
    n_epochs,
    lr,
    width_penalty_lambda=1e-3,
    width_max_threshold=300.0,
    width_min_threshold=20.0,
):
    n_bins = spectra.shape[1]
    frequencies = torch.arange(n_bins, dtype=torch.float32)
    model = GaussianNetwork(frequencies=frequencies, n_gaussians=n_gaussians)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    sigma_threshold_max = float(width_max_threshold / 2.355)
    sigma_threshold_min = float(width_min_threshold / 2.355)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(temperatures)
        mse_loss = criterion(y_pred, spectra)

        # Penalize width outside [min, max]
        sigma_list = []
        A_list = []
        for g in range(model.n_gaussians):
            sigma_list.append(torch.exp(model.sigma_nets[g](temperatures)))
            A_list.append(torch.sigmoid(model.A_nets[g](temperatures)))
        sigma_tensor = torch.cat(sigma_list, dim=1)
        A_tensor = torch.cat(A_list, dim=1)
        width_penalty_max = (torch.relu(sigma_tensor - temperatures.new_tensor(sigma_threshold_max))** 2).mean()
        width_penalty_min = (torch.relu(temperatures.new_tensor(sigma_threshold_min) - sigma_tensor)** 2).mean()
        loss = mse_loss + width_penalty_lambda * (width_penalty_max + width_penalty_min)
            
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            width_penalty = (width_penalty_lambda * (width_penalty_max + width_penalty_min)).item()
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, MSE: {mse_loss.item():.6f}, Width Penalty: {width_penalty:.6f}")

    return model, losses

def plot_dataset_modeling_results(model, df, temps_norm):
    spectra = np.stack(df["spectrum"].values)
    n_bins = spectra.shape[1]
    f_np = model.frequencies.numpy()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(float(f_np.min()), float(1000))
    ax.set_ylim(float(spectra.min()), float(spectra.max()))
    ax.set_xlabel("Frequency", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
    ax.grid(True)

    line_true, = ax.plot([], [], "b-", linewidth=1, label='True')
    line_pred, = ax.plot([], [], "r-", linewidth=1, label='Predicted')
    colors = ['orange', 'green', 'blue', 'violet', 'brown', 'magenta']
    gauss_lines = [ax.plot([], [], "--", color=colors[g % len(colors)], alpha=0.6, linewidth=1, label=f'Gaussian {g+1}')[0]
                   for g in range(model.n_gaussians)]
    temp_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=14)
    ax.legend(loc='upper right')

    def update(i):
        line_true.set_data(f_np, spectra[i])

        with torch.no_grad():
            y_pred = model(temps_norm[i:i+1]).squeeze().numpy()

            A_list, mu_list, sigma_list = [], [], []
            for g in range(model.n_gaussians):
                A_list.append(torch.sigmoid(model.A_nets[g](temps_norm[i:i+1])))
                mu_list.append(model.base_mu + model.mu_nets[g](temps_norm[i:i+1]))
                sigma_list.append(torch.exp(model.sigma_nets[g](temps_norm[i:i+1])))
            A = torch.cat(A_list, dim=1)
            mu = torch.cat(mu_list, dim=1)
            sigma = torch.cat(sigma_list, dim=1)

            f_plot = model.frequencies.view(1, -1)
            indiv = []
            for g in range(model.n_gaussians):
                gaussian = A[0, g] * torch.exp(-0.5 * ((f_plot - mu[0, g]) / sigma[0, g])**2)
                indiv.append(gaussian.squeeze().numpy())

        line_pred.set_data(f_np, y_pred)
        for g, gl in enumerate(gauss_lines):
            gl.set_data(f_np, indiv[g])

        temp = float(df["temperature"].iloc[i])
        temp_text.set_text(f"Temperature: {temp:.1f} °C")
        return (line_true, line_pred, *gauss_lines, temp_text)

    animation = FuncAnimation(fig, update, frames=spectra.shape[0], interval=100, blit=True)
    os.makedirs("plots", exist_ok=True)
    path = "plots/experimental_spectra_model.gif"
    animation.save(path, writer="pillow")
    plt.close(fig)
    show_image(path)

if __name__ == "__main__":
    # time_array = np.linspace(0, 4, 24*4)
    # _, temperatures, spectra = generate_synthetic_data_g3(time_array)
    # x = torch.linspace(0, 2027, 2028, dtype=torch.float32)
    # temps = torch.tensor(temperatures, dtype=torch.float32).unsqueeze(1)
    # spectra = torch.tensor(spectra, dtype=torch.float32)
    # temps_norm, temp_mean, temp_std = normalize_temperatures(temps)
    # print("Training model.")
    # model, losses = train_model(x, temps_norm, spectra, 3, 4000)
    # print("Plotting results.")
    # plot_sythetic_fitting_results(x, temps_norm, temps, spectra, model, losses)

    print("Training on experimental data.")
    df = prepare_ml_dataset()
    spectra_np = np.stack(df["spectrum"].values)
    temps = torch.tensor(df["temperature"].values, dtype=torch.float32).unsqueeze(1)
    spectra = torch.tensor(spectra_np, dtype=torch.float32)
    temps_norm, _, _ = normalize_temperatures(temps)
    model, losses = train_model(temps_norm, spectra, n_gaussians=3, n_epochs=10000, lr=0.01)

    print("Saving loss curve.")
    save_loss_curve(losses, path='plots/dataset_training_loss.png')

    print("Creating fit animation.")
    plot_dataset_modeling_results(model, df, temps_norm)
