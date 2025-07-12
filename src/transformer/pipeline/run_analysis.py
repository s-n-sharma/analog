# 📁 run_analysis.py (with Reconstruction Plot)

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

from pipeline import config
from pipeline.utils import load_model_and_normalization
from signal_decomposition.data.generation import generate_signal1, generate_signal2
from pipeline.cutoff_predictor.predictor import CutoffPredictor


# --- NEW HELPER FUNCTIONS for Ideal Bode Plots ---
def generate_ideal_lowpass_bode(freqs_hz, cutoff_hz):
    """Generates an ideal 1st-order low-pass Bode plot magnitude in dB."""
    omega = 2 * np.pi * freqs_hz
    omega_c = 2 * np.pi * cutoff_hz
    magnitude = 1 / np.sqrt(1 + (omega / omega_c)**2)
    return 20 * np.log10(magnitude)

def generate_ideal_highpass_bode(freqs_hz, cutoff_hz):
    """Generates an ideal 1st-order high-pass Bode plot magnitude in dB."""
    omega = 2 * np.pi * freqs_hz
    omega_c = 2 * np.pi * cutoff_hz
    ratio = np.divide(omega, omega_c, out=np.full_like(omega, 1e-9), where=(omega_c != 0))
    magnitude = ratio / np.sqrt(1 + ratio**2)
    return 20 * np.log10(magnitude)
# --- END NEW FUNCTIONS ---


def main():
    """
    Main pipeline to run signal decomposition, analysis, and plotting.
    """
    # 1. Generate Input Signal (Unchanged)
    print("Generating input signals...")
    lowpass_true_input = generate_signal1(config.FC_LOWPASS, config.SEQ_LEN)
    highpass_true_input = generate_signal2(config.FC_HIGHPASS, config.SEQ_LEN)
    mixed_signal = lowpass_true_input + highpass_true_input

    # 2. Load Signal Decomposition Model and Predict Components (Unchanged)
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

    # 3. Use the CNN Predictor to Find Cutoff Frequencies (Unchanged)
    print("Initializing cutoff frequency predictors...")
    try:
        lp_predictor = CutoffPredictor(model_type='lowpass')
        hp_predictor = CutoffPredictor(model_type='highpass')

        print("Predicting cutoff frequencies with trained CNNs...")
        lp_fc_pred = lp_predictor.predict(lowpass_pred_curve)
        hp_fc_pred = hp_predictor.predict(highpass_pred_curve)
        
        print(f"CNN Predicted Low-Pass Cutoff: {lp_fc_pred:.1f} Hz")
        print(f"CNN Predicted High-Pass Cutoff: {hp_fc_pred:.1f} Hz")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}\nCannot proceed with analysis. Exiting.")
        return

    # --- 4. Generate Ground Truth and Reconstructed Plots ---
    print("Generating comparison plots...")
    freq_output = np.logspace(2, 5, config.UPSAMPLED_LEN)

    # Generate the ideal Bode plots from the CNN's predicted frequencies
    ideal_lp_from_pred = generate_ideal_lowpass_bode(freq_output, lp_fc_pred)
    ideal_hp_from_pred = generate_ideal_highpass_bode(freq_output, hp_fc_pred)
    reconstructed_bandpass = ideal_lp_from_pred + ideal_hp_from_pred

    # Generate the ground truth band-pass curve using the original known frequencies
    ground_truth_lp = generate_ideal_lowpass_bode(freq_output, config.FC_LOWPASS)
    ground_truth_hp = generate_ideal_highpass_bode(freq_output, config.FC_HIGHPASS)
    ground_truth_bandpass = ground_truth_lp + ground_truth_hp
    # --- END NEW SECTION ---

    # 5. Plotting Results
    # Plot 1: The original decomposition plot
    plt.figure(figsize=(14, 7))
    plt.semilogx(freq_output, lowpass_pred_curve, 'b-', label='Predicted Low-Pass Response')
    plt.semilogx(freq_output, highpass_pred_curve, 'r-', label='Predicted High-Pass Response')
    plt.axvline(lp_fc_pred, color='b', linestyle='--', label=f'LP Cutoff (CNN): {lp_fc_pred:.1f} Hz')
    plt.axvline(hp_fc_pred, color='r', linestyle='--', label=f'HP Cutoff (CNN): {hp_fc_pred:.1f} Hz')
    plt.title('Filter Decomposition and CNN-Predicted Cutoff Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'cnn_cutoff_analysis.png'), dpi=300)
    
    # Plot 2: Reconstruction Comparison Plot
    plt.figure(figsize=(14, 7))
    plt.semilogx(freq_output, ground_truth_bandpass, 'k-', linewidth=2.5, label=f'Ground Truth Band-Pass (fc={config.FC_LOWPASS}/{config.FC_HIGHPASS} Hz)')
    plt.semilogx(freq_output, reconstructed_bandpass, 'g--', linewidth=2, label=f'Reconstructed Ideal Band-Pass (fc={lp_fc_pred:.1f}/{hp_fc_pred:.1f} Hz)')
    plt.title('Comparison of Ground Truth vs. Reconstructed Band-Pass Filter')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'reconstruction_comparison.png'), dpi=300)

    # --- Plot 3: MODIFIED - Ground Truth Input vs. Predicted Output ---
    plt.figure(figsize=(14, 7))
    
    # Define the frequency axis for the low-resolution input signals
    # NOTE: We assume the generation functions use the same frequency range.
    freq_input = np.logspace(2, 5, config.SEQ_LEN)

    # Plot the high-resolution predicted curves from the model (solid lines)
    plt.semilogx(freq_output, lowpass_pred_curve, 'b-', label='Predicted LP Curve (High-Res)')
    plt.semilogx(freq_output, highpass_pred_curve, 'r-', label='Predicted HP Curve (High-Res)')

    # Plot the low-resolution ground truth signals that were generated (dashed lines with markers)
    # These are the actual target signals for the model, just at a lower resolution.
    plt.semilogx(freq_input, lowpass_true_input, 'c--', marker='.', markersize=4, label=f'Ground Truth Input LP (Low-Res)')
    plt.semilogx(freq_input, highpass_true_input, color='orange', linestyle='--', marker='.', markersize=4, label=f'Ground Truth Input HP (Low-Res)')
    
    plt.title('Comparison of Ground Truth Input vs. Predicted Output Curves')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pipeline_dir, 'ground_truth_vs_predicted_components.png'), dpi=300)
    # --- END MODIFIED PLOT ---

    plt.show()


if __name__ == "__main__":
    main()
