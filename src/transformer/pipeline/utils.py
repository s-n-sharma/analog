import os
import numpy as np
import torch
from scipy.optimize import brentq

# Important: We need to tell Python where to find the signal_decomposition module
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from signal_decomposition.model import SignalDecompositionTransformer

def load_model_and_normalization(model_path, norm_path, model_params):
    """
    Loads the trained model and normalization parameters from specified paths.
    """
    print("Loading model and normalization parameters...")
    model = SignalDecompositionTransformer(
        seq_len=model_params['seq_len'],
        in_channels=model_params['in_channels'],
        d_model=model_params['d_model'],
        nhead=model_params['nhead'],
        num_layers=model_params['num_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        out_channels=model_params['out_channels'],
        upsampled_len=model_params['upsampled_len']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    norm_data = np.load(norm_path)
    norm_params = {
        'X_mean': float(norm_data['X_mean']),
        'X_std': float(norm_data['X_std']),
        'Y_mean': float(norm_data['Y_mean']),
        'Y_std': float(norm_data['Y_std'])
    }
    print("Model loaded successfully.")
    return model, norm_params

def find_cutoff(freqs, response_db, kind='lowpass'):
    """
    Finds the -3dB cutoff frequency of a filter response using brentq root finding.

    Args:
        freqs (np.array): Array of frequency points.
        response_db (np.array): Array of filter response in dB.
        kind (str): 'lowpass' or 'highpass'. Determines the reference point.

    Returns:
        float: The cutoff frequency in Hz, or None if not found.
    """
    # The reference is the maximum gain in the passband.
    ref = np.max(response_db)
    
    # This function will be 0 at the frequency where response is 3dB below the reference.
    def func_to_find_root(f):
        # Interpolate to get a more accurate value at frequency 'f'
        # np.interp is more robust than finding the closest index.
        return np.interp(f, freqs, response_db) - (ref - 3)

    # To use brentq, we need a bracket [a, b] where func(a) and func(b) have opposite signs.
    # We check the signs at the start and end of our frequency range.
    start_freq, end_freq = freqs[1], freqs[-1]
    
    try:
        # Check if a root actually exists in the given range
        if np.sign(func_to_find_root(start_freq)) == np.sign(func_to_find_root(end_freq)):
            # print(f"Warning: No cutoff found for '{kind}' in the given frequency range.")
            return None
        
        fc = brentq(func_to_find_root, start_freq, end_freq)
        return fc
    except ValueError as e:
        # This can happen if the signs are the same, which we already checked, but for safety.
        # print(f"Could not find cutoff for '{kind}': {e}")
        return None