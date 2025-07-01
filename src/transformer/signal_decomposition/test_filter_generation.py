import numpy as np
import matplotlib.pyplot as plt
from data.generation import (
    generate_signal1, generate_signal2, generate_sample, 
    generate_random_dataset, generate_bandpass_filter, 
    generate_notch_filter, generate_filter_with_phase
)

def test_filter_generation():
    """Test the new filter-based signal generation."""
    print("Testing filter-based signal generation...")
    
    # Test basic filter responses
    length = 1000
    fc_low = 1000  # 1 kHz cutoff for lowpass
    fc_high = 10000  # 10 kHz cutoff for highpass
    
    # Generate individual filter responses
    lowpass_response = generate_signal1(fc_low, length)
    highpass_response = generate_signal2(fc_high, length)
    
    # Generate a sample with mixed response
    lowpass, highpass, mixed = generate_sample(fc_low, fc_high, length)
    
    # Generate frequency axis for plotting
    freq = np.logspace(-1, 6, length)
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Individual filter responses
    plt.subplot(2, 3, 1)
    plt.semilogx(freq, lowpass_response, 'b-', label=f'Lowpass (fc={fc_low} Hz)')
    plt.semilogx(freq, highpass_response, 'r-', label=f'Highpass (fc={fc_high} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Individual Filter Responses')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Mixed response
    plt.subplot(2, 3, 2)
    plt.semilogx(freq, mixed, 'g-', label='Mixed Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Mixed Filter Response')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Comparison
    plt.subplot(2, 3, 3)
    plt.semilogx(freq, lowpass, 'b--', alpha=0.7, label='Lowpass Component')
    plt.semilogx(freq, highpass, 'r--', alpha=0.7, label='Highpass Component')
    plt.semilogx(freq, mixed, 'g-', linewidth=2, label='Mixed = LP + HP')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Component Decomposition')
    plt.legend()
    plt.grid(True)
    
    # Test additional filter types
    # Subplot 4: Bandpass filter
    plt.subplot(2, 3, 4)
    bandpass = generate_bandpass_filter(1000, 10000, length)
    plt.semilogx(freq, bandpass, 'm-', label='Bandpass (1kHz - 10kHz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Bandpass Filter Response')
    plt.legend()
    plt.grid(True)
    
    # Subplot 5: Notch filter
    plt.subplot(2, 3, 5)
    notch = generate_notch_filter(5000, 10, length)
    plt.semilogx(freq, notch, 'c-', label='Notch (fc=5kHz, Q=10)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Notch Filter Response')
    plt.legend()
    plt.grid(True)
    
    # Subplot 6: Phase response
    plt.subplot(2, 3, 6)
    mag_lp, phase_lp = generate_filter_with_phase(fc_low, length, 'lowpass')
    mag_hp, phase_hp = generate_filter_with_phase(fc_high, length, 'highpass')
    plt.semilogx(freq, np.degrees(phase_lp), 'b-', label='Lowpass Phase')
    plt.semilogx(freq, np.degrees(phase_hp), 'r-', label='Highpass Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (degrees)')
    plt.title('Filter Phase Responses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('filter_responses_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated {length} frequency points from {freq[0]:.1f} Hz to {freq[-1]:.0f} Hz")
    print(f"Lowpass response range: {lowpass_response.min():.3f} to {lowpass_response.max():.3f}")
    print(f"Highpass response range: {highpass_response.min():.3f} to {highpass_response.max():.3f}")
    print(f"Mixed response range: {mixed.min():.3f} to {mixed.max():.3f}")

def test_random_dataset():
    """Test random dataset generation with filter responses."""
    print("\nTesting random dataset generation...")
    
    # Generate a small random dataset
    N = 5
    length = 500
    lowpass_data, highpass_data, mixed_data = generate_random_dataset(
        N, length, 
        fc1_range=(100, 5000),  # Lowpass cutoffs: 100Hz to 5kHz
        fc2_range=(1000, 50000), # Highpass cutoffs: 1kHz to 50kHz
        seed=42
    )
    
    print(f"Generated dataset shapes:")
    print(f"  Lowpass: {lowpass_data.shape}")
    print(f"  Highpass: {highpass_data.shape}")
    print(f"  Mixed: {mixed_data.shape}")
    
    # Plot a few examples
    freq = np.logspace(-1, 6, length)
    plt.figure(figsize=(12, 8))
    
    for i in range(min(3, N)):
        plt.subplot(2, 3, i*2 + 1)
        plt.semilogx(freq, lowpass_data[i], 'b-', alpha=0.7, label='Lowpass')
        plt.semilogx(freq, highpass_data[i], 'r-', alpha=0.7, label='Highpass')
        plt.semilogx(freq, mixed_data[i], 'g-', linewidth=2, label='Mixed')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Sample {i+1} - Individual Components')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, i*2 + 2)
        plt.semilogx(freq, mixed_data[i], 'g-', linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Sample {i+1} - Mixed Response')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('random_dataset_test.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_filter_generation()
    test_random_dataset()
    print("\nFilter-based signal generation testing complete!")
