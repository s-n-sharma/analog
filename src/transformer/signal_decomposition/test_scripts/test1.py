import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import torch
from model.transformer import SignalDecompositionTransformer
from data.generation import generate_signal1, generate_signal2

def load_model_and_normalization():
    """Load the trained model and normalization parameters."""
    model = SignalDecompositionTransformer(
        seq_len=128, in_channels=1, d_model=768, nhead=12, 
        num_layers=4, dim_feedforward=2048, out_channels=2, upsampled_len=512
    )
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'output_models', 'signal_decomposition_transformer.pth')
    norm_path = os.path.join(os.path.dirname(__file__), '..', 'output_models', 'normalization_params.npz')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ Trained model loaded successfully")
    else:
        print("⚠️  Model file not found - using untrained model for demo")
    
    # Load normalization parameters
    norm_params = None
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path)
        norm_params = {
            'X_mean': float(norm_data['X_mean']),
            'X_std': float(norm_data['X_std']),
            'Y_mean': float(norm_data['Y_mean']),
            'Y_std': float(norm_data['Y_std'])
        }
        print("✅ Normalization parameters loaded")
        print(f"   Input: mean={norm_params['X_mean']:.2f}, std={norm_params['X_std']:.2f}")
        print(f"   Output: mean={norm_params['Y_mean']:.2f}, std={norm_params['Y_std']:.2f}")
    else:
        print("⚠️  Normalization parameters not found - using raw values")
    
    model.eval()
    return model, norm_params

def main_demo():
    """
    Main demonstration showing:
    - Top row: Two input curves and their mixed signal
    - Bottom row: Model predictions of the decomposed signals
    """
    print("🎯 Filter Decomposition Transformer Demo")
    print("=" * 45)
    
    # Load the trained model and normalization parameters
    print("📥 Loading model and normalization...")
    model, norm_params = load_model_and_normalization()
    
    # Generate example filter responses
    print("🔧 Generating filter responses...")
    fc_lowpass = 1000   # 1 kHz lowpass cutoff
    fc_highpass = 5000  # 5 kHz highpass cutoff
    seq_len = 128
    
    # Create the input signals
    lowpass_curve = generate_signal1(fc_lowpass, seq_len)
    highpass_curve = generate_signal2(fc_highpass, seq_len)
    mixed_signal = lowpass_curve + highpass_curve
    
    print(f"   Lowpass filter: fc = {fc_lowpass} Hz")
    print(f"   Highpass filter: fc = {fc_highpass} Hz")
    
    # Get model predictions
    print("🤖 Running model prediction...")
    x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # Apply input normalization if available
    if norm_params:
        x_input_norm = (x_input - norm_params['X_mean']) / norm_params['X_std']
    else:
        x_input_norm = x_input
        print("⚠️  Using unnormalized input")
    
    with torch.no_grad():
        predictions = model(x_input_norm)  # Shape: (1, 2, 512)
        
        # Denormalize predictions if normalization was used
        if norm_params:
            predictions = predictions * norm_params['Y_std'] + norm_params['Y_mean']
        
        lowpass_predicted = predictions[0, 0, :].numpy()
        highpass_predicted = predictions[0, 1, :].numpy()
    
    print("✅ Prediction complete")
    
    # Create frequency axes
    freq_input = np.logspace(-1, 6, seq_len)  # Input frequency range
    freq_output = np.logspace(-1, 6, 512)     # Model output frequency range
    
    # Create the main visualization
    print("📊 Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # === TOP ROW: INPUT DATA ===
    # Plot 1: Lowpass curve
    axes[0, 0].semilogx(freq_input, lowpass_curve, 'b-', linewidth=4, alpha=0.8)
    axes[0, 0].set_title(f'Lowpass Filter Response\n(fc = {fc_lowpass} Hz)', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.4)
    axes[0, 0].set_ylim([-60, 5])  # Typical dB range for filters
    axes[0, 0].set_xlim([1e-1, 1e6])
    
    # Plot 2: Highpass curve
    axes[0, 1].semilogx(freq_input, highpass_curve, 'r-', linewidth=4, alpha=0.8)
    axes[0, 1].set_title(f'Highpass Filter Response\n(fc = {fc_highpass} Hz)', 
                        fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].set_ylim([-60, 5])  # Typical dB range for filters
    axes[0, 1].set_xlim([1e-1, 1e6])
    
    # Plot 3: Mixed signal
    axes[0, 2].semilogx(freq_input, mixed_signal, 'g-', linewidth=4, alpha=0.9)
    axes[0, 2].set_title('Mixed Signal\n(Lowpass + Highpass)', 
                        fontsize=14, fontweight='bold', color='darkgreen', pad=15)
    axes[0, 2].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[0, 2].grid(True, alpha=0.4)
    axes[0, 2].set_ylim([-60, 10])  # Mixed signal can be slightly higher
    axes[0, 2].set_xlim([1e-1, 1e6])
    
    # === BOTTOM ROW: MODEL PREDICTIONS ===
    # Plot 4: Predicted lowpass
    axes[1, 0].semilogx(freq_output, lowpass_predicted, 'b--', linewidth=4, alpha=0.8)
    axes[1, 0].set_title('Model Predicted Lowpass', 
                        fontsize=14, fontweight='bold', color='darkblue', pad=15)
    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1, 0].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.4)
    axes[1, 0].set_ylim([-60, 5])
    axes[1, 0].set_xlim([1e-1, 1e6])
    
    # Plot 5: Predicted highpass
    axes[1, 1].semilogx(freq_output, highpass_predicted, 'r--', linewidth=4, alpha=0.8)
    axes[1, 1].set_title('Model Predicted Highpass', 
                        fontsize=14, fontweight='bold', color='darkred', pad=15)
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1, 1].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.4)
    axes[1, 1].set_ylim([-60, 5])
    axes[1, 1].set_xlim([1e-1, 1e6])
    
    # Plot 6: Reconstructed mixed signal
    reconstructed_mixed = lowpass_predicted + highpass_predicted
    axes[1, 2].semilogx(freq_output, reconstructed_mixed, 'purple', linewidth=4, alpha=0.8)
    axes[1, 2].set_title('Model Reconstructed Mixed', 
                        fontsize=14, fontweight='bold', color='purple', pad=15)
    axes[1, 2].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[1, 2].set_ylabel('Magnitude (dB)', fontsize=12)
    axes[1, 2].grid(True, alpha=0.4)
    axes[1, 2].set_ylim([-60, 10])
    axes[1, 2].set_xlim([1e-1, 1e6])
    
    # Add row labels on the left
    fig.text(0.02, 0.75, 'INPUT\nSIGNALS', fontsize=18, fontweight='bold', 
             rotation=90, verticalalignment='center', horizontalalignment='center')
    fig.text(0.02, 0.25, 'MODEL\nPREDICTIONS', fontsize=18, fontweight='bold', 
             rotation=90, verticalalignment='center', horizontalalignment='center', color='darkblue')
    
    # Main title
    plt.suptitle('Filter Response Decomposition using Transformer Neural Network', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1)
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'filter_decomposition_main_demo.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Saved main demo plot: {output_path}")
    
    plt.show()
    
    # Print analysis
    print("\n📈 ANALYSIS:")
    print(f"   Input mixed signal range: [{mixed_signal.min():.3f}, {mixed_signal.max():.3f}]")
    print(f"   Predicted lowpass range: [{lowpass_predicted.min():.3f}, {lowpass_predicted.max():.3f}]")
    print(f"   Predicted highpass range: [{highpass_predicted.min():.3f}, {highpass_predicted.max():.3f}]")
    
    # Calculate reconstruction error
    from torch.nn.functional import interpolate
    mixed_upsampled = interpolate(
        torch.tensor(mixed_signal).unsqueeze(0).unsqueeze(0), 
        size=512, mode='linear', align_corners=False
    )[0, 0].numpy()
    
    mse_error = np.mean((reconstructed_mixed - mixed_upsampled)**2)
    print(f"   Reconstruction MSE: {mse_error:.6f}")
    
    if mse_error < 0.01:
        print("   ✅ Excellent reconstruction quality!")
    elif mse_error < 0.1:
        print("   ✅ Good reconstruction quality")
    else:
        print("   ⚠️  Reconstruction could be improved")

if __name__ == "__main__":
    main_demo()
    print("\n🎉 Demo complete!")
    print("This demonstration shows how the transformer model takes a mixed")
    print("filter response and decomposes it into lowpass and highpass components.")
