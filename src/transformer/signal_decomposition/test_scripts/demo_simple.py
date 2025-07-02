import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model.transformer import SignalDecompositionTransformer
from data.generation import generate_signal1, generate_signal2, generate_sample

def load_model():
    """Load the trained transformer model and normalization parameters."""
    model = SignalDecompositionTransformer(
        seq_len=128, in_channels=1, d_model=768, nhead=12, 
        num_layers=4, dim_feedforward=2048, out_channels=2, upsampled_len=512
    )
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'output_models', 'signal_decomposition_transformer.pth')
    norm_path = os.path.join(os.path.dirname(__file__), '..', 'output_models', 'normalization_params.npz')
    
    norm_params = None
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✓ Trained model loaded")
    else:
        print("⚠ Using untrained model (for demo)")
    
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        norm_params = {k: float(v) for k, v in norm_data.items()}
        print("✓ Normalization parameters loaded")
    else:
        print("⚠ No normalization parameters found")
    
    model.eval()
    return model, norm_params

def demonstrate_filter_decomposition():
    """Main demonstration function."""
    
    # Generate example filter responses
    fc_lowpass = 1000   # 1 kHz
    fc_highpass = 5000  # 5 kHz
    seq_len = 128
    
    # Create ground truth signals
    lowpass_curve = generate_signal1(fc_lowpass, seq_len)
    highpass_curve = generate_signal2(fc_highpass, seq_len)
    mixed_signal = lowpass_curve + highpass_curve
    
    # Frequency axis
    freq = np.logspace(-1, 6, seq_len)
    freq_model = np.logspace(-1, 6, 512)  # Model outputs at higher resolution
    
    # Get model predictions
    model, norm_params = load_model()
    x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # Apply normalization if available
    if norm_params:
        x_input = (x_input - norm_params['X_mean']) / norm_params['X_std']
    
    with torch.no_grad():
        predictions = model(x_input)  # Shape: (1, 2, 512)
        
        # Denormalize if needed
        if norm_params:
            predictions = predictions * norm_params['Y_std'] + norm_params['Y_mean']
            
        lowpass_pred = predictions[0, 0, :].numpy()
        highpass_pred = predictions[0, 1, :].numpy()
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # TOP ROW: Input curves and mixed signal
    # Plot 1: Lowpass curve
    axes[0, 0].semilogx(freq, lowpass_curve, 'b-', linewidth=3, label=f'Lowpass (fc={fc_lowpass} Hz)')
    axes[0, 0].set_title('Input: Lowpass Filter', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([-60, 5])
    
    # Plot 2: Highpass curve  
    axes[0, 1].semilogx(freq, highpass_curve, 'r-', linewidth=3, label=f'Highpass (fc={fc_highpass} Hz)')
    axes[0, 1].set_title('Input: Highpass Filter', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([-60, 5])
    
    # Plot 3: Mixed signal
    axes[0, 2].semilogx(freq, mixed_signal, 'g-', linewidth=3, label='Mixed = LP + HP')
    axes[0, 2].set_title('Input: Mixed Signal', fontsize=14, fontweight='bold', color='darkgreen')
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('Magnitude (dB)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_ylim([-60, 10])
    
    # BOTTOM ROW: Model predictions
    # Plot 4: Model predicted lowpass
    axes[1, 0].semilogx(freq_model, lowpass_pred, 'b--', linewidth=3, alpha=0.8, label='Model Prediction')
    axes[1, 0].set_title('Model Output: Lowpass', fontsize=14, fontweight='bold', color='darkblue')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude (dB)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim([-60, 5])
    
    # Plot 5: Model predicted highpass
    axes[1, 1].semilogx(freq_model, highpass_pred, 'r--', linewidth=3, alpha=0.8, label='Model Prediction')
    axes[1, 1].set_title('Model Output: Highpass', fontsize=14, fontweight='bold', color='darkred')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude (dB)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([-60, 5])
    
    # Plot 6: Model reconstructed mixed signal
    reconstructed = lowpass_pred + highpass_pred
    axes[1, 2].semilogx(freq_model, reconstructed, 'purple', linewidth=3, alpha=0.8, label='Model Reconstruction')
    axes[1, 2].set_title('Model Output: Reconstructed Mixed', fontsize=14, fontweight='bold', color='purple')
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('Magnitude (dB)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].set_ylim([-60, 10])
    
    # Add row labels
    fig.text(0.02, 0.75, 'INPUT', fontsize=16, fontweight='bold', rotation=90, 
             verticalalignment='center', color='black')
    fig.text(0.02, 0.25, 'MODEL\nOUTPUT', fontsize=16, fontweight='bold', rotation=90, 
             verticalalignment='center', color='darkblue')
    
    plt.suptitle('Filter Response Decomposition using Transformer', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, top=0.93)
    
    # Save and show
    output_path = os.path.join(os.path.dirname(__file__), 'filter_decomposition_demo.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.show()
    
    # Print some stats
    print(f"\nFilter Parameters:")
    print(f"  Lowpass cutoff: {fc_lowpass} Hz")
    print(f"  Highpass cutoff: {fc_highpass} Hz")
    print(f"  Input resolution: {seq_len} points")
    print(f"  Output resolution: {len(lowpass_pred)} points")

if __name__ == "__main__":
    print("🎯 Filter Decomposition Demonstration")
    print("=" * 40)
    demonstrate_filter_decomposition()
    print("✅ Done!")
