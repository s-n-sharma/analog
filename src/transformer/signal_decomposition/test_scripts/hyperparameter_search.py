import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from model.transformer import SignalDecompositionTransformer
from data.generation import generate_random_dataset

def quick_training_test(config):
    """Quick training test with given configuration."""
    print(f"Testing config: {config}")
    
    # Generate small dataset for quick test
    lowpass, highpass, mixed = generate_random_dataset(1000, 128, seed=42)
    
    # Normalize data
    X = torch.tensor(mixed, dtype=torch.float32).unsqueeze(-1)
    Y = torch.cat([
        torch.tensor(lowpass, dtype=torch.float32).unsqueeze(1),
        torch.tensor(highpass, dtype=torch.float32).unsqueeze(1)
    ], dim=1)
    
    X_norm = (X - X.mean()) / X.std()
    Y_norm = (Y - Y.mean()) / Y.std()
    
    # Create model
    model = SignalDecompositionTransformer(
        seq_len=128, in_channels=1, out_channels=2, upsampled_len=128,  # No upsampling for quick test
        d_model=config['d_model'], nhead=config['nhead'], 
        num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward']
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    # Quick training
    model.train()
    losses = []
    for epoch in range(10):  # Just 10 epochs for quick test
        optimizer.zero_grad()
        pred = model(X_norm)
        loss = loss_fn(pred, Y_norm)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # Check convergence
    improvement = losses[0] - losses[-1]
    final_loss = losses[-1]
    
    print(f"  Final loss: {final_loss:.6f}, Improvement: {improvement:.6f}")
    return final_loss, improvement

def find_best_config():
    """Find best hyperparameter configuration."""
    print("🔍 Hyperparameter Search")
    print("=" * 30)
    
    configs = [
        # Smaller models
        {'d_model': 256, 'nhead': 8, 'num_layers': 2, 'dim_feedforward': 1024, 'lr': 1e-3},
        {'d_model': 512, 'nhead': 8, 'num_layers': 2, 'dim_feedforward': 2048, 'lr': 5e-4},
        
        # Medium models
        {'d_model': 512, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 2048, 'lr': 5e-4},
        {'d_model': 768, 'nhead': 12, 'num_layers': 3, 'dim_feedforward': 2048, 'lr': 3e-4},
        
        # Larger models
        {'d_model': 768, 'nhead': 12, 'num_layers': 4, 'dim_feedforward': 3072, 'lr': 1e-4},
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}/{len(configs)}:")
        try:
            final_loss, improvement = quick_training_test(config)
            results.append((config, final_loss, improvement))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((config, float('inf'), 0))
    
    # Sort by final loss
    results.sort(key=lambda x: x[1])
    
    print(f"\n🏆 BEST CONFIGURATIONS:")
    for i, (config, loss, improvement) in enumerate(results[:3]):
        print(f"{i+1}. Loss: {loss:.6f}, Improvement: {improvement:.6f}")
        print(f"   Config: {config}")
    
    return results[0][0]  # Return best config

def print_training_tips():
    """Print tips for better training."""
    print(f"\n💡 TRAINING TIPS:")
    print("1. Delete old cache: rm signal_decomp_data_cache.npz")
    print("2. Use the improved architecture with:")
    print("   - More layers (4-6)")
    print("   - Larger model size (768-1024 d_model)")
    print("   - Lower learning rate (1e-4 to 5e-4)")
    print("   - More training data (10k+ samples)")
    print("   - More epochs (50-100)")
    print("3. Monitor validation loss - stop if overfitting")
    print("4. Use frequency-aware loss function")
    print("5. Add data augmentation (noise, frequency shifts)")

if __name__ == "__main__":
    best_config = find_best_config()
    print_training_tips()
    
    print(f"\n🎯 RECOMMENDED CONFIG:")
    print(f"   {best_config}")
    print("\nUse this configuration in your train.py script for better results!")
