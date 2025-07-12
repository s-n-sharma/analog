# 📁 cutoff_predictor/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

from model import CNN1D
from data_generator import generate_filter_data, create_dataloaders

# --- Training Configuration ---
NUM_SAMPLES = 30000
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.002
MODEL_SAVE_DIR = "trained_models"

def train_model(filter_type='lowpass'):
    """
    Trains a CNN model to predict the cutoff frequency for a given filter type.
    """
    print(f"--- Starting training for {filter_type.upper()} model ---")
    
    # 1. Prepare Data
    print("Generating dataset...")
    X, y = generate_filter_data(NUM_SAMPLES, filter_type=filter_type)
    train_loader, val_loader = create_dataloaders(X, y, batch_size=BATCH_SIZE)
    
    # 2. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNN1D(output_size=1).to(device)
    criterion = nn.MSELoss() # Mean Squared Error is good for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 4. Save the trained model
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    model_path = os.path.join(MODEL_SAVE_DIR, f"{filter_type}_cutoff_predictor.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return history

def plot_history(history, filter_type):
    """Plots the training and validation loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss History for {filter_type.upper()} Model')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'{filter_type}_loss_history.png'))
    plt.show()


if __name__ == "__main__":
    # Train the low-pass model
    lp_history = train_model(filter_type='lowpass')
    plot_history(lp_history, 'lowpass')
    
    # Train the high-pass model
    hp_history = train_model(filter_type='highpass')
    plot_history(hp_history, 'highpass')
