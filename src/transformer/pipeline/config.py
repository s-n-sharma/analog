# 📁 config.py

import os

# --- File Paths ---
# Assumes this script is in a directory at the same level as 'signal_decomposition'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'signal_decomposition', 'output_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'signal_decomposition_transformer.pth')
NORM_PATH = os.path.join(MODEL_DIR, 'normalization_params.npz')

# --- Signal Generation Parameters ---
SEQ_LEN = 128
UPSAMPLED_LEN = 512
FC_LOWPASS = 1000   # Cutoff frequency for the low-pass signal component
FC_HIGHPASS = 10000  # Cutoff frequency for the high-pass signal component

# --- Model Parameters (must match the trained model) ---
D_MODEL = 768
NHEAD = 12
NUM_LAYERS = 4
DIM_FEEDFORWARD = 2048
IN_CHANNELS = 1
OUT_CHANNELS = 2