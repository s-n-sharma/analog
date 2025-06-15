import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os


NUM_SAMPLES = 50000
NUM_FREQ_POINTS = 100
FUNCTION_NAME = "RC_lowpass"
DATA_FILE_PATH = f"./{FUNCTION_NAME}_data.npz"

# Training
BATCH_SIZE = 64
LEARNING_RATE = 1e-3 
EPOCHS = 40
VALIDATION_SPLIT = 0.2

class FunctionalDataGeneration:
    """Class to efficiently generate synthetic data for a given functional circuit."""
    
    def __init__(self, function_name, num_samples, num_freq_points, output_path):
        self.function_name = function_name
        self.num_samples = num_samples
        self.num_freq_points = num_freq_points
        self.output_path = output_path
        # Frequency range for simulation (log scale is better for Bode plots)
        self.frequency_range = np.logspace(2, 6, num=self.num_freq_points) # 100 Hz to 1 MHz
        
    def low_pass_response(self, freqs, omega0):
        """Calculates the magnitude response for a simple RC low-pass filter."""
        # Using angular frequency for calculation
        omega = 2 * np.pi * freqs
        return np.abs(1 / (1 + 1j * (omega / omega0)))

    def generate_data(self):
        """
        Generates a synthetic dataset. The key change is to sample omega0 directly
        in a range relevant to the frequency axis.
        """
        print("Generating synthetic data with corrected methodology...")
        X_data = np.zeros((self.num_samples, self.num_freq_points), dtype=np.float32)
        y_data = np.zeros((self.num_samples, 1), dtype=np.float32)
        
        min_omega_range = 2 * np.pi * self.frequency_range.min()
        max_omega_range = 2 * np.pi * self.frequency_range.max()
        
        log_omega0_values = np.random.uniform(np.log(min_omega_range), np.log(max_omega_range), self.num_samples)
        omega0_values = np.exp(log_omega0_values)

        for i in range(self.num_samples):
            omega0 = omega0_values[i]
            mags = self.low_pass_response(self.frequency_range, omega0)
            
            mags_db = 20 * np.log10(mags + 1e-8)
            
            noise = np.random.normal(0, 0.1, mags_db.shape) # Noise on dB scale
            
            X_data[i, :] = mags_db + noise
            y_data[i, 0] = omega0
            
        np.savez(self.output_path, X=X_data, y=y_data)
        print(f"Data saved at: {self.output_path}")
        return self.output_path

# --- 2. PyTorch Dataset ---
class FunctionalDataset(Dataset):
    """Custom dataset, now handles dB-scaled features."""
    def __init__(self, data_path, transform_labels=True):
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        features_np = data['X'].astype(np.float32)
        labels_np = data['y'].astype(np.float32)
        
        # Add a channel dimension for the CNN: (N, C, L) -> (N, 1, 100)
        self.features = torch.from_numpy(features_np).unsqueeze(1)
        self.labels_raw = torch.from_numpy(labels_np)
        self.transform_labels = transform_labels

        self.feature_mean = self.features.mean(dim=[0, 2], keepdim=True)
        self.feature_std = self.features.std(dim=[0, 2], keepdim=True)
        self.features = (self.features - self.feature_mean) / (self.feature_std + 1e-8)

        if self.transform_labels:
            self.labels = torch.log(self.labels_raw)
        else:
            self.labels = self.labels_raw

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 3. CNN Model (The Major Upgrade) ---
class CNN1D(nn.Module):
    """
    A 1D Convolutional Neural Network. This is much better at finding patterns
    (like the 'knee' of the filter) in sequence data than a simple MLP.
    """
    def __init__(self, output_size):
        super(CNN1D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, NUM_FREQ_POINTS)
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.shape[1] * dummy_output.shape[2]

        self.dense_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x

# --- 4. Main Training Script ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_FILE_PATH):
        fdg = FunctionalDataGeneration(FUNCTION_NAME, NUM_SAMPLES, NUM_FREQ_POINTS, DATA_FILE_PATH)
        fdg.generate_data()
    
    full_dataset = FunctionalDataset(DATA_FILE_PATH)
    
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Initialize CNN Model ---
    model = CNN1D(output_size=1).to(device)
    print("Using 1D CNN Model:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_rc_model_cnn.pth")
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.6f}")

    print("\nTraining complete.")