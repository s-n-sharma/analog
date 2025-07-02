import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1D signals."""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class SignalDecompositionTransformer(nn.Module):
    """
    Improved transformer for decomposing mixed frequency responses into component filters.
    Includes better upsampling and frequency-aware processing.
    """
    def __init__(self, seq_len=128, in_channels=1, d_model=768, nhead=12, num_layers=4, dim_feedforward=2048, out_channels=2, upsampled_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.upsampled_len = upsampled_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Improved input embedding with multiple conv layers
        self.input_embed = nn.Sequential(
            nn.Conv1d(in_channels, d_model//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Improved upsampling - gradual upsampling with smoothing
        if upsampled_len > seq_len:
            upsample_factor = upsampled_len // seq_len
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(d_model, d_model, kernel_size=upsample_factor, stride=upsample_factor),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),  # Smoothing
                nn.BatchNorm1d(d_model),
                nn.ReLU()
            )
        else:
            self.upsample = nn.Identity()
        
        # Output projection with residual connection
        self.proj = nn.Sequential(
            nn.Conv1d(d_model, d_model//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model//2, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        x = self.input_embed(x)  # (batch, d_model, seq_len)
        x = x.transpose(1, 2)   # (batch, seq_len, d_model)
        x = self.pos_encoder(x) # (batch, seq_len, d_model)
        x = self.transformer(x) # (batch, seq_len, d_model)
        x = x.transpose(1, 2)   # (batch, d_model, seq_len)
        x = self.upsample(x)    # (batch, d_model, upsampled_len)
        x = self.proj(x)        # (batch, out_channels, upsampled_len)
        return x