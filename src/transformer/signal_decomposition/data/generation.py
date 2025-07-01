import numpy as np
from typing import Callable, Tuple

def generate_signal1(cutoff_freq: float, length: int) -> np.ndarray:
    """Low pass filter response: H(jω) = 1 / (1 + jω/ωc)"""
    # Generate frequency range from 10^-1 to 10^6 Hz (log scale)
    freq = np.logspace(-1, 6, length)
    # Normalized angular frequency (ω/ωc)
    normalized_freq = freq / cutoff_freq
    # Low pass filter magnitude response: |H(jω)| = 1 / sqrt(1 + (ω/ωc)²)
    magnitude = 1 / np.sqrt(1 + normalized_freq**2)
    return magnitude

def generate_signal2(cutoff_freq: float, length: int) -> np.ndarray:
    """High pass filter response: H(jω) = (jω/ωc) / (1 + jω/ωc)"""
    # Generate frequency range from 10^-1 to 10^6 Hz (log scale)
    freq = np.logspace(-1, 6, length)
    # Normalized angular frequency (ω/ωc)
    normalized_freq = freq / cutoff_freq
    # High pass filter magnitude response: |H(jω)| = (ω/ωc) / sqrt(1 + (ω/ωc)²)
    magnitude = normalized_freq / np.sqrt(1 + normalized_freq**2)
    return magnitude

def generate_sample(
    fc1: float, fc2: float, length: int,
    f1: Callable[[float, int], np.ndarray] = generate_signal1,
    f2: Callable[[float, int], np.ndarray] = generate_signal2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (lowpass_response, highpass_response, mixed_signal) for given cutoff frequencies."""
    lowpass = f1(fc1, length)
    highpass = f2(fc2, length)
    # Mix the signals with some overlap in frequency domain
    mixed = lowpass + highpass
    return lowpass, highpass, mixed

def generate_dataset(
    fc1_vals, fc2_vals, length: int,
    f1: Callable[[float, int], np.ndarray] = generate_signal1,
    f2: Callable[[float, int], np.ndarray] = generate_signal2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate arrays of (lowpass_response, highpass_response, mixed_signal) for all cutoff frequencies."""
    lowpass_list, highpass_list, mixed_list = [], [], []
    for fc1 in fc1_vals:
        for fc2 in fc2_vals:
            lowpass, highpass, mixed = generate_sample(fc1, fc2, length, f1, f2)
            lowpass_list.append(lowpass)
            highpass_list.append(highpass)
            mixed_list.append(mixed)
    return np.stack(lowpass_list), np.stack(highpass_list), np.stack(mixed_list)

def generate_random_dataset(
    N: int, length: int,
    f1: Callable[[float, int], np.ndarray] = generate_signal1,
    f2: Callable[[float, int], np.ndarray] = generate_signal2,
    fc1_range=(10, 10000), fc2_range=(100, 100000), seed=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate N random (lowpass_response, highpass_response, mixed_signal) samples with cutoff frequencies in given ranges."""
    rng = np.random.default_rng(seed)
    # Generate cutoff frequencies in log scale for more realistic distribution
    fc1_vals = np.exp(rng.uniform(np.log(fc1_range[0]), np.log(fc1_range[1]), N))
    fc2_vals = np.exp(rng.uniform(np.log(fc2_range[0]), np.log(fc2_range[1]), N))
    lowpass_list, highpass_list, mixed_list = [], [], []
    for fc1, fc2 in zip(fc1_vals, fc2_vals):
        lowpass, highpass, mixed = generate_sample(fc1, fc2, length, f1, f2)
        lowpass_list.append(lowpass)
        highpass_list.append(highpass)
        mixed_list.append(mixed)
    return np.stack(lowpass_list), np.stack(highpass_list), np.stack(mixed_list)

def generate_bandpass_filter(fc_low: float, fc_high: float, length: int) -> np.ndarray:
    """Generate a bandpass filter response between fc_low and fc_high."""
    freq = np.logspace(-1, 6, length)
    # Bandpass is combination of high pass and low pass
    normalized_freq_low = freq / fc_low
    normalized_freq_high = freq / fc_high
    # High pass component
    hp_magnitude = normalized_freq_low / np.sqrt(1 + normalized_freq_low**2)
    # Low pass component  
    lp_magnitude = 1 / np.sqrt(1 + normalized_freq_high**2)
    # Bandpass is the product
    magnitude = hp_magnitude * lp_magnitude
    return magnitude

def generate_notch_filter(fc_center: float, Q: float, length: int) -> np.ndarray:
    """Generate a notch (band-stop) filter response centered at fc_center with quality factor Q."""
    freq = np.logspace(-1, 6, length)
    # Normalized frequency
    omega_n = freq / fc_center
    # Notch filter transfer function magnitude
    magnitude = np.abs((omega_n**2 + 1) / (omega_n**2 + omega_n/Q + 1))
    return magnitude

def add_noise_to_signal(signal: np.ndarray, noise_level: float = 0.01, seed=None) -> np.ndarray:
    """Add Gaussian noise to a signal."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level, signal.shape)
    return signal + noise

def generate_filter_with_phase(cutoff_freq: float, length: int, filter_type: str = 'lowpass') -> Tuple[np.ndarray, np.ndarray]:
    """Generate filter response with both magnitude and phase information."""
    freq = np.logspace(-1, 6, length)
    normalized_freq = freq / cutoff_freq
    
    if filter_type == 'lowpass':
        # Low pass filter
        magnitude = 1 / np.sqrt(1 + normalized_freq**2)
        phase = -np.arctan(normalized_freq)  # Phase in radians
    elif filter_type == 'highpass':
        # High pass filter
        magnitude = normalized_freq / np.sqrt(1 + normalized_freq**2)
        phase = np.pi/2 - np.arctan(normalized_freq)  # Phase in radians
    else:
        raise ValueError("filter_type must be 'lowpass' or 'highpass'")
    
    return magnitude, phase