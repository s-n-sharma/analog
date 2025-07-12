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
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
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
    else:
        print("⚠️  Normalization parameters not found - using raw values")
    
    model.eval()
    return model, norm_params

def test_multiple_frequencies():
    """Test the model with multiple frequency combinations and analyze quality."""
    print("🔧 Testing Multiple Frequency Combinations")
    print("=" * 50)
    
    model, norm_params = load_model_and_normalization()
    
    # Test cases with different frequency separations
    test_cases = [
        (500, 2000, "Close frequencies"),
        (1000, 10000, "1 decade separation"), 
        (100, 50000, "Wide separation"),
        (2000, 5000, "Medium separation"),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 4, figsize=(20, 4*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    errors = []
    
    for i, (fc_lp, fc_hp, description) in enumerate(test_cases):
        # Generate test signals
        seq_len = 128
        lowpass_true = generate_signal1(fc_lp, seq_len)
        highpass_true = generate_signal2(fc_hp, seq_len)
        mixed_signal = lowpass_true + highpass_true
        
        # Model prediction with normalization
        x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        if norm_params:
            x_input_norm = (x_input - norm_params['X_mean']) / norm_params['X_std']
        else:
            x_input_norm = x_input
        
        with torch.no_grad():
            predictions = model(x_input_norm)
            
            if norm_params:
                predictions = predictions * norm_params['Y_std'] + norm_params['Y_mean']
            
            lowpass_pred = predictions[0, 0, :].numpy()
            highpass_pred = predictions[0, 1, :].numpy()
        
        # Frequency axes
        freq_input = np.logspace(-1, 6, seq_len)
        freq_output = np.logspace(-1, 6, 512)
        
        # Calculate errors
        # Upsample ground truth for comparison
        from torch.nn.functional import interpolate
        lowpass_true_up = interpolate(
            torch.tensor(lowpass_true).unsqueeze(0).unsqueeze(0), 
            size=512, mode='linear', align_corners=False
        )[0, 0].numpy()
        highpass_true_up = interpolate(
            torch.tensor(highpass_true).unsqueeze(0).unsqueeze(0), 
            size=512, mode='linear', align_corners=False
        )[0, 0].numpy()
        
        lp_error = np.mean(np.abs(lowpass_pred - lowpass_true_up))
        hp_error = np.mean(np.abs(highpass_pred - highpass_true_up))
        errors.append((lp_error, hp_error))
        
        # Plot results
        # Original mixed signal
        axes[i, 0].semilogx(freq_input, mixed_signal, 'g-', linewidth=2)
        axes[i, 0].set_title(f'{description}\nMixed Signal (LP: {fc_lp}Hz, HP: {fc_hp}Hz)')
        axes[i, 0].set_ylabel('Magnitude (dB)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylim([-70, 10])
        
        # Ground truth components
        axes[i, 1].semilogx(freq_input, lowpass_true, 'b-', linewidth=2, label='True LP')
        axes[i, 1].semilogx(freq_input, highpass_true, 'r-', linewidth=2, label='True HP')
        axes[i, 1].set_title('Ground Truth Components')
        axes[i, 1].set_ylabel('Magnitude (dB)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim([-70, 10])
        
        # Predicted components
        axes[i, 2].semilogx(freq_output, lowpass_pred, 'b--', linewidth=2, alpha=0.8, label='Pred LP')
        axes[i, 2].semilogx(freq_output, highpass_pred, 'r--', linewidth=2, alpha=0.8, label='Pred HP')
        axes[i, 2].set_title(f'Model Predictions\nLP Error: {lp_error:.1f}dB, HP Error: {hp_error:.1f}dB')
        axes[i, 2].set_ylabel('Magnitude (dB)')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_ylim([-70, 10])
        
        # Comparison overlay
        axes[i, 3].semilogx(freq_output, lowpass_true_up, 'b-', linewidth=1, alpha=0.7, label='True LP')
        axes[i, 3].semilogx(freq_output, lowpass_pred, 'b--', linewidth=2, alpha=0.9, label='Pred LP')
        axes[i, 3].semilogx(freq_output, highpass_true_up, 'r-', linewidth=1, alpha=0.7, label='True HP')
        axes[i, 3].semilogx(freq_output, highpass_pred, 'r--', linewidth=2, alpha=0.9, label='Pred HP')
        axes[i, 3].set_title('True vs Predicted Overlay')
        axes[i, 3].set_xlabel('Frequency (Hz)')
        axes[i, 3].set_ylabel('Magnitude (dB)')
        axes[i, 3].legend()
        axes[i, 3].grid(True, alpha=0.3)
        axes[i, 3].set_ylim([-70, 10])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), 'improved_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved analysis to: {output_path}")
    plt.show()
    
    # Print summary
    print(f"\n📊 QUALITY ANALYSIS:")
    avg_lp_error = np.mean([e[0] for e in errors])
    avg_hp_error = np.mean([e[1] for e in errors])
    print(f"   Average Lowpass Error: {avg_lp_error:.2f} dB")
    print(f"   Average Highpass Error: {avg_hp_error:.2f} dB")
    
    if avg_lp_error < 5.0 and avg_hp_error < 5.0:
        print("   ✅ Good prediction quality!")
    elif avg_lp_error < 10.0 and avg_hp_error < 10.0:
        print("   ⚠️  Moderate prediction quality")
    else:
        print("   ❌ Poor prediction quality - consider retraining")

if __name__ == "__main__":
    test_multiple_frequencies()
    print("\n✅ Improved analysis complete!")
