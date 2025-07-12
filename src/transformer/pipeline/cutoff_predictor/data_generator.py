# 📁 cutoff_predictor/data_generator.py

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# This must match the model's expected input size
NUM_FREQ_POINTS = 512 
# Frequency range for our generated data
FREQ_RANGE_HZ = np.logspace(2, 5, NUM_FREQ_POINTS) # 100 Hz to 100 kHz

def generate_filter_data(num_samples, filter_type='lowpass'):
    """
    Generates a dataset of ideal filter responses and their cutoff frequencies.
    
    Args:
        num_samples (int): The number of filter curves to generate.
        filter_type (str): 'lowpass' or 'highpass'.

    Returns:
        Tuple[np.array, np.array]: A tuple of (filter_responses, log_cutoff_freqs).
    """
    # X data: the filter curves
    responses = np.zeros((num_samples, NUM_FREQ_POINTS))
    # y data: the cutoff frequencies (we will store their log)
    log_cutoffs = np.zeros(num_samples)
    
    # Define the range for random cutoff frequencies (e.g., 500 Hz to 50 kHz)
    min_fc = 500
    max_fc = 50000

    for i in range(num_samples):
        # Generate a random cutoff frequency within the range
        fc = np.random.uniform(min_fc, max_fc)
        log_cutoffs[i] = np.log10(fc)
        
        # Generate the ideal Bode plot for this cutoff frequency
        omega = 2 * np.pi * FREQ_RANGE_HZ
        omega_c = 2 * np.pi * fc
        
        if filter_type == 'lowpass':
            magnitude = 1 / np.sqrt(1 + (omega / omega_c)**2)
        elif filter_type == 'highpass':
            ratio = np.divide(omega, omega_c, out=np.full_like(omega, 1e-9), where=(omega_c != 0))
            magnitude = ratio / np.sqrt(1 + ratio**2)
        else:
            raise ValueError("filter_type must be 'lowpass' or 'highpass'")
            
        # Convert to dB and add a small amount of noise to make training more robust
        responses[i, :] = 20 * np.log10(magnitude) + np.random.normal(0, 0.1, NUM_FREQ_POINTS)

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
