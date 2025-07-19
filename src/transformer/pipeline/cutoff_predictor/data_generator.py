# 📁 cutoff_predictor/data_generator.py (Improved)

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# input size 
NUM_FREQ_POINTS = 512 
FREQ_RANGE_HZ = np.logspace(2, 5, NUM_FREQ_POINTS)

def generate_filter_data(num_samples, filter_type='lowpass'):
    """
    Generates a dataset of ideal filter responses with added shape distortion
    and noise for more robust training.
    
    Args:
        num_samples (int): The number of filter curves to generate.
        filter_type (str): 'lowpass' or 'highpass'.

    Returns:
        Tuple[np.array, np.array]: A tuple of (filter_responses, log_cutoff_freqs).
    """
    responses = np.zeros((num_samples, NUM_FREQ_POINTS))
    log_cutoffs = np.zeros(num_samples)
    min_fc = 500
    max_fc = 50000

    for i in range(num_samples):
        fc = np.random.uniform(min_fc, max_fc)
        log_cutoffs[i] = np.log10(fc)
        
        # Generate the ideal 1st-order Bode plot
        omega = 2 * np.pi * FREQ_RANGE_HZ
        omega_c = 2 * np.pi * fc
        
        if filter_type == 'lowpass':
            magnitude = 1 / np.sqrt(1 + (omega / omega_c)**2)
        elif filter_type == 'highpass':
            ratio = np.divide(omega, omega_c, out=np.full_like(omega, 1e-9), where=(omega_c != 0))
            magnitude = ratio / np.sqrt(1 + ratio**2)
        else:
            raise ValueError("filter_type must be 'lowpass' or 'highpass'")
        
        # Convert to dB
        response_db = 20 * np.log10(np.maximum(magnitude, 1e-5))

        # shape distortion 
        distortion_amplitude = np.random.uniform(0.5, 2.0) # dB
        distortion_frequency = np.random.uniform(0.5, 2.0)
        distortion_phase = np.random.uniform(0, 2 * np.pi)
        
        # Create a distortion wave over the log-spaced frequency points
        distortion = distortion_amplitude * np.sin(
            distortion_frequency * np.linspace(0, 2 * np.pi, NUM_FREQ_POINTS) + distortion_phase
        )
        
        # Add the distortion and some small random noise
        responses[i, :] = response_db #+ distortion + np.random.normal(0, 0.2, NUM_FREQ_POINTS)

    # Reshape for the CNN (batch, channels, length)
    responses = responses.reshape(-1, 1, NUM_FREQ_POINTS)
    return responses.astype(np.float32), log_cutoffs.astype(np.float32)

def create_dataloaders(X, y, batch_size=32):
    """Creates training and validation dataloaders from the data."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    # Split into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
