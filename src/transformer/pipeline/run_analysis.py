# 📁 run_analysis.py (with Unit-Corrected -3dB Analysis)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Robust Project Path Configuration ---
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(pipeline_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our project modules
from pipeline import config
from pipeline.utils import load_model_and_normalization, find_cutoff
from signal_decomposition.data.generation import generate_signal1, generate_signal2

# --- Helper Functions for Ideal Bode Plots (Matching Data Generation Logic) ---
def generate_ideal_lowpass_bode(freqs_hz, cutoff_hz):
    """
    Generates an ideal 1st-order low-pass Bode plot magnitude in dB.
    This function's logic is identical to the project's generate_signal1.
    """
    normalized_freq = freqs_hz / cutoff_hz
    magnitude = 1 / np.sqrt(1 + normalized_freq**2)
    return 20 * np.log10(np.maximum(magnitude, 1e-9))

def generate_ideal_highpass_bode(freqs_hz, cutoff_hz):
    """
    Generates an ideal 1st-order high-pass Bode plot magnitude in dB.
    This function's logic is identical to the project's generate_signal2.
    """
    normalized_freq = freqs_hz / cutoff_hz
    magnitude = normalized_freq / np.sqrt(1 + normalized_freq**2)
    return 20 * np.log10(np.maximum(magnitude, 1e-9))
# --- End Helper Functions ---


def main():
    """
    Main pipeline to run signal decomposition and test the -3dB cutoff method
    with unit-corrected logic.
    """
    # 1. Generate Input Signal & Run Transformer
    print("Generating input signals...")
    lowpass_true_input = generate_signal1(config.FC_LOWPASS, config.SEQ_LEN)
    highpass_true_input = generate_signal2(config.FC_HIGHPASS, config.SEQ_LEN)
    mixed_signal = lowpass_true_input + highpass_true_input

    model_params = {
        'seq_len': config.SEQ_LEN, 'in_channels': config.IN_CHANNELS, 'd_model': config.D_MODEL,
        'nhead': config.NHEAD, 'num_layers': config.NUM_LAYERS, 'dim_feedforward': config.DIM_FEEDFORWARD,
        'out_channels': config.OUT_CHANNELS, 'upsampled_len': config.UPSAMPLED_LEN
    }
    model, norm_params = load_model_and_normalization(config.MODEL_PATH, config.NORM_PATH, model_params)
    x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    x_input_norm = (x_input - norm_params['X_mean']) / norm_params['X_std']
    print("Running signal decomposition model...")
    with torch.no_grad():
        predictions = model(x_input_norm)
        predictions = predictions * norm_params['Y_std'] + norm_params['Y_mean']
        lowpass_pred_curve = predictions[0, 0, :].numpy()
        highpass_pred_curve = predictions[0, 1, :].numpy()

    # 2. Find Cutoff Frequencies using the -3dB Method
    # The frequency axis must match the one used during generation/prediction
    freq_output = np.logspace(-1, 6, config.UPSAMPLED_LEN)
    
    print("\n--- Calculating with -3dB rule ---")
    lp_fc_3db = find_cutoff(freq_output, lowpass_pred_curve, kind='lowpass')
    hp_fc_3db = find_cutoff(freq_output, highpass_pred_curve, kind='highpass')
    print(f"-3dB Predicted Low-Pass Cutoff: {lp_fc_3db:.1f} Hz" if lp_fc_3db else "Low-Pass Cutoff not found.")
    print(f"-3dB Predicted High-Pass Cutoff: {hp_fc_3db:.1f} Hz" if hp_fc_3db else "High-Pass Cutoff not found.")

    # 3. Generate Curves for Plotting
    # Generate the ground truth curves for comparison
    ground_truth_lp = generate_ideal_lowpass_bode(freq_output, config.FC_LOWPASS)
    ground_truth_hp = generate_ideal_highpass_bode(freq_output, config.FC_HIGHPASS)
    ground_truth_bandpass = ground_truth_lp + ground_truth_hp
    
    # Generate ideal curves based on the -3dB method's findings
    reconstructed_lp_3db = generate_ideal_lowpass_bode(freq_output, lp_fc_3db) if lp_fc_3db else None
    reconstructed_hp_3db = generate_ideal_highpass_bode(freq_output, hp_fc_3db) if hp_fc_3db else None
    
    # Sum the reconstructed components to get the reconstructed band-pass filter
    if reconstructed_lp_3db is not None and reconstructed_hp_3db is not None:
        reconstructed_bandpass_3db = reconstructed_lp_3db + reconstructed_hp_3db
    else:
        reconstructed_bandpass_3db = None

    # 4. Create Comparison Plots
    print("\nGenerating comparison plots...")
    
    # --- Plot 1: Low-Pass and High-Pass Analysis ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    # Subplot 1: Low-Pass Comparison
    ax1.semilogx(freq_output, ground_truth_lp, 'c--', linewidth=2.5, label=f'Ground Truth (fc={config.FC_LOWPASS:.0f} Hz)')
    ax1.semilogx(freq_output, lowpass_pred_curve, 'b-', label='Transformer Prediction')
    if reconstructed_lp_3db is not None:
        ax1.semilogx(freq_output, reconstructed_lp_3db, 'b:', linewidth=2, label=f'Ideal Curve from -3dB (fc={lp_fc_3db:.0f} Hz)')
    ax1.set_title('Low-Pass Filter Analysis')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    # Subplot 2: High-Pass Comparison
    ax2.semilogx(freq_output, ground_truth_hp, 'm--', linewidth=2.5, label=f'Ground Truth (fc={config.FC_HIGHPASS:.0f} Hz)')
    ax2.semilogx(freq_output, highpass_pred_curve, 'r-', label='Transformer Prediction')
    if reconstructed_hp_3db is not None:
        ax2.semilogx(freq_output, reconstructed_hp_3db, 'r:', linewidth=2, label=f'Ideal Curve from -3dB (fc={hp_fc_3db:.0f} Hz)')
    ax2.set_title('High-Pass Filter Analysis')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    
    fig.suptitle('Analysis of -3dB Cutoff Method on Transformer Output', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(pipeline_dir, '3db_method_verification.png'), dpi=300)
    
    # --- Plot 2: Reconstructed Band-Pass Comparison ---
    plt.figure(figsize=(14, 8))
    plt.semilogx(freq_output, ground_truth_bandpass, 'k-', linewidth=2.5, label=f'Ground Truth Band-Pass')
    if reconstructed_bandpass_3db is not None:
        plt.semilogx(freq_output, reconstructed_bandpass_3db, 'g--', linewidth=2, label=f'Reconstructed Band-Pass from -3dB')
    plt.title('Comparison of Ground Truth vs. Reconstructed Band-Pass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'reconstruction_comparison.png'), dpi=300)

    plt.show()

if __name__ == "__main__":
    main()
