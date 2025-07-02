import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from model.transformer import SignalDecompositionTransformer
from data.generation import generate_signal1, generate_signal2

def load_model():
    """Load the trained model."""
    model = SignalDecompositionTransformer(
        seq_len=128, in_channels=1, d_model=768, nhead=12, 
        num_layers=4, dim_feedforward=2048, out_channels=2, upsampled_len=512
    )
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'output_models', 'signal_decomposition_transformer.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✓ Model loaded successfully")
    else:
        print("⚠ Model file not found - using untrained model")
    
    model.eval()
    return model

def test_custom_filters(fc_lowpass=1000, fc_highpass=5000):
    """
    Test the model with custom filter parameters.
    
    Args:
        fc_lowpass: Cutoff frequency for lowpass filter (Hz)
        fc_highpass: Cutoff frequency for highpass filter (Hz)
    """
    print(f"\n🔧 Testing with LP={fc_lowpass} Hz, HP={fc_highpass} Hz")
    
    # Generate signals
    seq_len = 128
    lowpass_true = generate_signal1(fc_lowpass, seq_len)
    highpass_true = generate_signal2(fc_highpass, seq_len)
    mixed_signal = lowpass_true + highpass_true
    
    # Model prediction
    model = load_model()
    x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        pred = model(x_input)
        lowpass_pred = pred[0, 0, :].numpy()
        highpass_pred = pred[0, 1, :].numpy()
    
    # Frequency axes
    freq_input = np.logspace(-1, 6, seq_len)
    freq_output = np.logspace(-1, 6, 512)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: INPUT DATA
    axes[0, 0].semilogx(freq_input, lowpass_true, 'b-', linewidth=3)
    axes[0, 0].set_title(f'INPUT: Lowpass Filter\n(fc = {fc_lowpass} Hz)', fontweight='bold')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([-60, 5])
    
    axes[0, 1].semilogx(freq_input, highpass_true, 'r-', linewidth=3)
    axes[0, 1].set_title(f'INPUT: Highpass Filter\n(fc = {fc_highpass} Hz)', fontweight='bold')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-60, 5])
    
    axes[0, 2].semilogx(freq_input, mixed_signal, 'g-', linewidth=3)
    axes[0, 2].set_title('INPUT: Mixed Signal\n(Lowpass + Highpass)', fontweight='bold', color='darkgreen')
    axes[0, 2].set_ylabel('Magnitude (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-60, 10])
    
    # Row 2: MODEL PREDICTIONS
    axes[1, 0].semilogx(freq_output, lowpass_pred, 'b--', linewidth=3, alpha=0.8)
    axes[1, 0].set_title('MODEL: Predicted Lowpass', fontweight='bold', color='darkblue')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-60, 5])
    
    axes[1, 1].semilogx(freq_output, highpass_pred, 'r--', linewidth=3, alpha=0.8)
    axes[1, 1].set_title('MODEL: Predicted Highpass', fontweight='bold', color='darkred')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-60, 5])
    
    # Reconstructed mixed signal
    reconstructed = lowpass_pred + highpass_pred
    axes[1, 2].semilogx(freq_output, reconstructed, 'purple', linewidth=3, alpha=0.8)
    axes[1, 2].set_title('MODEL: Reconstructed Mixed', fontweight='bold', color='purple')
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('Magnitude (dB)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([-60, 10])
    
    # Add side labels
    fig.text(0.02, 0.75, 'INPUT\nDATA', fontsize=14, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.25, 'MODEL\nOUTPUT', fontsize=14, fontweight='bold', 
             rotation=90, va='center', ha='center', color='darkblue')
    
    plt.suptitle(f'Filter Decomposition Demo (LP: {fc_lowpass} Hz, HP: {fc_highpass} Hz)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.92)
    
    # Save with parameter info in filename
    filename = f'demo_LP{fc_lowpass}_HP{fc_highpass}.png'
    output_path = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {filename}")
    
    plt.show()
    
    return lowpass_pred, highpass_pred, reconstructed

def run_multiple_tests():
    """Run tests with different filter combinations."""
    print("🚀 Running Multiple Filter Tests")
    print("=" * 40)
    
    test_cases = [
        (500, 2000),    # Low frequencies
        (1000, 5000),   # Medium frequencies  
        (2000, 10000),  # Higher frequencies
        (100, 50000),   # Wide separation
    ]
    
    for fc_lp, fc_hp in test_cases:
        test_custom_filters(fc_lp, fc_hp)
        print("-" * 40)

if __name__ == "__main__":
    print("🎯 Interactive Filter Decomposition Demo")
    print("=" * 50)
    
    # Option 1: Run with default parameters
    print("\n1. Default test (1kHz LP + 5kHz HP):")
    test_custom_filters(1000, 5000)
    
    # Option 2: Custom parameters (modify these values)
    print("\n2. Custom test - MODIFY THESE VALUES:")
    custom_lp = 800   # Change this value
    custom_hp = 8000  # Change this value
    test_custom_filters(custom_lp, custom_hp)
    
    # Option 3: Multiple tests
    print("\n3. Multiple test cases:")
    # Uncomment the line below to run multiple tests
    # run_multiple_tests()
    
    print("\n✅ Demo complete!")
    print("\nTo test different frequencies, modify the custom_lp and custom_hp values in the script.")
