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
N_SAMPLES = 5000
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE_PATH = 'signal_decomp_data_cache.npz'
VAL_FRAC = 0.1  # 10% for validation

# Data loading/generation with caching
if os.path.exists(CACHE_PATH):
    print(f"Loading cached dataset from {CACHE_PATH}")
    arr = np.load(CACHE_PATH)
    lowpass, highpass, mixed = arr['s1'], arr['s2'], arr['mixed']
else:
    print(f"Generating new filter response dataset and caching to {CACHE_PATH}")
    lowpass, highpass, mixed = generate_random_dataset(N_SAMPLES, SEQ_LEN)
    np.savez(CACHE_PATH, s1=lowpass, s2=highpass, mixed=mixed)

# X: (N, SEQ_LEN, 1)
X = torch.tensor(mixed, dtype=torch.float32).unsqueeze(-1)
# Upsample targets to match model output length
lowpass_up = F.interpolate(torch.tensor(lowpass, dtype=torch.float32).unsqueeze(1), size=UPSAMPLED_LEN, mode='linear', align_corners=False)
highpass_up = F.interpolate(torch.tensor(highpass, dtype=torch.float32).unsqueeze(1), size=UPSAMPLED_LEN, mode='linear', align_corners=False)
# Y: (N, OUT_CHANNELS, UPSAMPLED_LEN)
Y = torch.cat([lowpass_up, highpass_up], dim=1)

dataset = TensorDataset(X, Y)

# Split into train/val
val_size = int(VAL_FRAC * N_SAMPLES)
train_size = N_SAMPLES - val_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = SignalDecompositionTransformer(
    seq_len=SEQ_LEN,
    in_channels=IN_CHANNELS,
    d_model=512,
    nhead=8,
    num_layers=2,
    dim_feedforward=1024,
    out_channels=OUT_CHANNELS,
    upsampled_len=UPSAMPLED_LEN
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
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
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'signal_decomposition_transformer.pth')
print("Model saved successfully! The transformer can now decompose mixed filter responses into lowpass and highpass components.") 