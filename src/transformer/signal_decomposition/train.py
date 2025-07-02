import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import os
import numpy as np

from model import SignalDecompositionTransformer
from data import generate_random_dataset

# Config
SEQ_LEN = 128
UPSAMPLED_LEN = 512
IN_CHANNELS = 1  # Only the mixed filter response as input
OUT_CHANNELS = 2  # Two filter responses to recover (lowpass + highpass)
N_SAMPLES = 10000  # Increased from 5000
BATCH_SIZE = 64    # Increased from 32
EPOCHS = 50        # Increased from 20
LR = 5e-4          # Reduced learning rate for more stable training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE_PATH = 'signal_decomp_data_cache.npz'
VAL_FRAC = 0.1  # 10% for validation

# Data loading/generation with caching
# NOTE: If you've updated the generation functions (e.g., switched to dB scale),
# delete the cache file to regenerate data with the new format
if os.path.exists(CACHE_PATH):
    print(f"Loading cached dataset from {CACHE_PATH}")
    print("NOTE: If generation format has changed, delete cache file to regenerate")
    arr = np.load(CACHE_PATH)
    lowpass, highpass, mixed = arr['s1'], arr['s2'], arr['mixed']
else:
    print(f"Generating new filter response dataset and caching to {CACHE_PATH}")
    lowpass, highpass, mixed = generate_random_dataset(N_SAMPLES, SEQ_LEN)
    np.savez(CACHE_PATH, s1=lowpass, s2=highpass, mixed=mixed)

# X: (N, SEQ_LEN, 1)
X = torch.tensor(mixed, dtype=torch.float32).unsqueeze(-1)

# Normalize the input data (mixed signal in dB)
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std
print(f"Input normalization - Mean: {X_mean:.3f} dB, Std: {X_std:.3f} dB")

# Upsample targets to match model output length
lowpass_up = F.interpolate(torch.tensor(lowpass, dtype=torch.float32).unsqueeze(1), size=UPSAMPLED_LEN, mode='linear', align_corners=False)
highpass_up = F.interpolate(torch.tensor(highpass, dtype=torch.float32).unsqueeze(1), size=UPSAMPLED_LEN, mode='linear', align_corners=False)

# Y: (N, OUT_CHANNELS, UPSAMPLED_LEN)
Y = torch.cat([lowpass_up, highpass_up], dim=1)

# Normalize the target data (individual filter responses in dB)
Y_mean = Y.mean()
Y_std = Y.std()
Y_normalized = (Y - Y_mean) / Y_std
print(f"Target normalization - Mean: {Y_mean:.3f} dB, Std: {Y_std:.3f} dB")

# Save normalization parameters for inference
normalization_params = {
    'X_mean': X_mean.item(), 'X_std': X_std.item(),
    'Y_mean': Y_mean.item(), 'Y_std': Y_std.item()
}
print(f"Data ranges - Input: [{X.min():.1f}, {X.max():.1f}] dB, Target: [{Y.min():.1f}, {Y.max():.1f}] dB")
print(f"Normalized ranges - Input: [{X_normalized.min():.3f}, {X_normalized.max():.3f}], Target: [{Y_normalized.min():.3f}, {Y_normalized.max():.3f}]")

dataset = TensorDataset(X_normalized, Y_normalized)

# Split into train/val - use actual dataset length to avoid rounding errors
dataset_len = len(dataset)
val_size = int(VAL_FRAC * dataset_len)
train_size = dataset_len - val_size
print(f"Dataset length: {dataset_len}, Train size: {train_size}, Val size: {val_size}")
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Model with improved architecture
model = SignalDecompositionTransformer(
    seq_len=SEQ_LEN,
    in_channels=IN_CHANNELS,
    d_model=768,        # Increased from 512
    nhead=12,           # Increased from 8
    num_layers=4,       # Increased from 2
    dim_feedforward=2048, # Increased from 1024
    out_channels=OUT_CHANNELS,
    upsampled_len=UPSAMPLED_LEN
).to(DEVICE)

# Improved optimizer with scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Better loss function for frequency domain
def frequency_aware_loss(pred, target):
    """Loss function that emphasizes low frequency accuracy and smoothness."""
    # Standard MSE loss
    mse_loss = nn.MSELoss()(pred, target)
    
    # Smoothness penalty (L2 of first derivative)
    pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
    target_diff = target[:, :, 1:] - target[:, :, :-1]
    smoothness_loss = nn.MSELoss()(pred_diff, target_diff)
    
    # Low frequency emphasis (first 25% of frequencies more important)
    low_freq_weight = 2.0
    freq_len = pred.size(-1)
    low_freq_end = freq_len // 4
    
    low_freq_loss = nn.MSELoss()(pred[:, :, :low_freq_end], target[:, :, :low_freq_end])
    
    total_loss = mse_loss + 0.1 * smoothness_loss + low_freq_weight * low_freq_loss
    return total_loss

loss_fn = frequency_aware_loss

# Training loop
print(f"\n🚀 Starting training on {DEVICE}")
print(f"📊 Dataset: {N_SAMPLES} samples, {train_size} train, {val_size} validation")
print(f"🔧 Model: {sum(p.numel() for p in model.parameters())} parameters")

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate
        total_loss += loss.item() * xb.size(0)
        batch_count += 1
    avg_loss = total_loss / train_size

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * xb.size(0)
    avg_val_loss = val_loss / val_size
    
    # Track best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} - Train Loss: {avg_loss:.6f} - Val Loss: {avg_val_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f} {'⭐' if avg_val_loss == best_val_loss else ''}")

print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

# Save model and normalization parameters
model_save_path = os.path.join('output_models', 'signal_decomposition_transformer.pth')
norm_save_path = os.path.join('output_models', 'normalization_params.npz')

# Create output directory if it doesn't exist
os.makedirs('output_models', exist_ok=True)

torch.save(model.state_dict(), model_save_path)
np.savez(norm_save_path, **normalization_params)

print(f"Model saved to: {model_save_path}")
print(f"Normalization parameters saved to: {norm_save_path}")
print("The transformer can now decompose mixed filter responses into lowpass and highpass components.")
print("Remember to use the same normalization when running inference!") 