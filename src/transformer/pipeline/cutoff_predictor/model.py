import torch
import torch.nn as nn

NUM_FREQ_POINTS = 512

import torch
import torch.nn as nn

NUM_FREQ_POINTS = 512

class CNN1D(nn.Module):
    """
    A more robust 1D Convolutional Neural Network to predict a single value 
    (e.g., cutoff frequency) from a 1D signal (e.g., a filter's frequency response).
    
    Improvements include Batch Normalization and increased layer capacity.
    """
    def __init__(self, output_size=1):
        super(CNN1D, self).__init__()
        
        # CNN based
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.flatten = nn.Flatten()
        
        # This block dynamically calculates the input size for the dense layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, NUM_FREQ_POINTS)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
            
        # Dense layers to regress to the final output value
        self.dense_layers = nn.Sequential(
            nn.Linear(flattened_size, 256), # Increased size
            nn.ReLU(),
            nn.Dropout(0.4), # Increased dropout slightly
            nn.Linear(256, 128), # Added another dense layer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x
