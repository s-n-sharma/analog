# 📁 cutoff_predictor/train.py (Improved)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

# --- Robust Path Setup ---
# This ensures we can import from the parent 'pipeline' and 'signal_decomposition' directories
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(pipeline_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules from the project
from model import CNN1D
from config import MODEL_SAVE_DIR
from pipeline import config as main_config
from pipeline.utils import load_model_and_normalization
from signal_decomposition.data.generation import generate_signal1, generate_signal2

# --- Training Configuration ---
NUM_SAMPLES = 20000  # Number of samples to generate for training the CNN
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

def create_cnn_training_data_from_transformer(num_samples):
    """
    Uses the main SignalDecompositionTransformer to generate a realistic training
    dataset for the CNN cutoff predictor.
    """
    print("Loading the main SignalDecompositionTransformer to generate training data...")
    model_params = {
        'seq_len': main_config.SEQ_LEN, 'in_channels': main_config.IN_CHANNELS, 'd_model': main_config.D_MODEL,
        'nhead': main_config.NHEAD, 'num_layers': main_config.NUM_LAYERS, 'dim_feedforward': main_config.DIM_FEEDFORWARD,
        'out_channels': main_config.OUT_CHANNELS, 'upsampled_len': main_config.UPSAMPLED_LEN
    }
    transformer_model, norm_params = load_model_and_normalization(main_config.MODEL_PATH, main_config.NORM_PATH, model_params)
    transformer_model.eval()
    
    # Pre-allocate arrays
    lp_curves = np.zeros((num_samples, 1, main_config.UPSAMPLED_LEN), dtype=np.float32)
    hp_curves = np.zeros((num_samples, 1, main_config.UPSAMPLED_LEN), dtype=np.float32)
    lp_log_fcs = np.zeros(num_samples, dtype=np.float32)
    hp_log_fcs = np.zeros(num_samples, dtype=np.float32)

    print(f"Generating {num_samples} data samples through the transformer...")
    for i in range(num_samples):
        # Generate random but separated cutoff frequencies
        fc_lp = np.random.uniform(500, 5000)
        fc_hp = np.random.uniform(fc_lp + 2000, 50000)
        
        # Store the log10 of the true cutoff frequency as the label
        lp_log_fcs[i] = np.log10(fc_lp)
        hp_log_fcs[i] = np.log10(fc_hp)

        # Generate the mixed signal input
        mixed_signal = generate_signal1(fc_lp, main_config.SEQ_LEN) + generate_signal2(fc_hp, main_config.SEQ_LEN)
        
        # Prepare input for the transformer
        x_input = torch.tensor(mixed_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        x_input_norm = (x_input - norm_params['X_mean']) / norm_params['X_std']
        
        # Get the transformer's output curves
        with torch.no_grad():
            predictions = transformer_model(x_input_norm)
            predictions = predictions * norm_params['Y_std'] + norm_params['Y_mean']
            lp_curves[i, 0, :] = predictions[0, 0, :].numpy()
            hp_curves[i, 0, :] = predictions[0, 1, :].numpy()
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{num_samples} samples...")

    return (lp_curves, lp_log_fcs), (hp_curves, hp_log_fcs)

def train_predictor(filter_type, data):
    """Trains a CNN predictor on the provided data."""
    X, y = data
    print(f"\n--- Starting training for {filter_type.upper()} predictor ---")
    
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), labels).item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss/len(val_loader):.6f}")

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, f"{filter_type}_cutoff_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # 1. Generate realistic training data using the transformer
    lp_data, hp_data = create_cnn_training_data_from_transformer(NUM_SAMPLES)
    
    # 2. Train the low-pass predictor on the transformer's LP outputs
    train_predictor('lowpass', lp_data)
    
    # 3. Train the high-pass predictor on the transformer's HP outputs
    train_predictor('highpass', hp_data)
