import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from data.generation import generate_random_dataset

def test_data_generation():
    """Test the dB-scale data generation and show statistics."""
    print("🔍 Testing dB-scale data generation...")
    
    # Generate a small dataset
    N = 100
    length = 128
    lowpass, highpass, mixed = generate_random_dataset(N, length, seed=42)
    
    print(f"\n📊 Dataset Statistics (N={N}, length={length}):")
    print(f"   Lowpass:  min={lowpass.min():.1f} dB, max={lowpass.max():.1f} dB, mean={lowpass.mean():.1f} dB, std={lowpass.std():.1f} dB")
    print(f"   Highpass: min={highpass.min():.1f} dB, max={highpass.max():.1f} dB, mean={highpass.mean():.1f} dB, std={highpass.std():.1f} dB")
    print(f"   Mixed:    min={mixed.min():.1f} dB, max={mixed.max():.1f} dB, mean={mixed.mean():.1f} dB, std={mixed.std():.1f} dB")
    
    # Plot some examples
    freq = np.logspace(-1, 6, length)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot first 3 examples
    for i in range(3):
        # Individual components
        axes[0, i].semilogx(freq, lowpass[i], 'b-', alpha=0.7, label='Lowpass')
        axes[0, i].semilogx(freq, highpass[i], 'r-', alpha=0.7, label='Highpass')
        axes[0, i].set_title(f'Sample {i+1}: Components')
        axes[0, i].set_ylabel('Magnitude (dB)')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim([-70, 10])
        
        # Mixed signal
        axes[1, i].semilogx(freq, mixed[i], 'g-', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: Mixed Signal')
        axes[1, i].set_xlabel('Frequency (Hz)')
        axes[1, i].set_ylabel('Magnitude (dB)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim([-70, 10])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'data_generation_test.png'), dpi=300)
    plt.show()
    
    # Test normalization
    print(f"\n🔧 Testing Normalization:")
    mixed_mean = mixed.mean()
    mixed_std = mixed.std()
    mixed_norm = (mixed - mixed_mean) / mixed_std
    
    print(f"   Original: mean={mixed_mean:.2f}, std={mixed_std:.2f}")
    print(f"   Normalized: mean={mixed_norm.mean():.6f}, std={mixed_norm.std():.6f}")
    print(f"   Range: [{mixed.min():.1f}, {mixed.max():.1f}] → [{mixed_norm.min():.3f}, {mixed_norm.max():.3f}]")
    
    return lowpass, highpass, mixed

if __name__ == "__main__":
    test_data_generation()
    print("\n✅ Data generation test complete!")
