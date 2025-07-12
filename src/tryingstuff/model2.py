import torch
import torch.nn as nn
import numpy as np
import math
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# --- Section 1: Constants and Configuration ---

# Signal and Model Hyperparameters
SIGNAL_LENGTH = 400
PATCH_SIZE = 20
D_MODEL = 256
N_HEAD = 16
N_LAYERS = 8

# Training Hyperparameters
EPOCHS = 20
LR = 0.0001
BATCH_SIZE = 32
itemcount = 0

# Loss Weight for the STOP token
STOP_LOSS_WEIGHT = 1.2

##################################################

def bandpass(params, length):
    r1, c1, r2, c2 = params[0], params[1], params[2], params[3]
    ret = []
    x = np.logspace(1, 6, length)
    for w in x:
        num = 1j * w * r2*c1
        denom = -w*w*r1*r2*c1*c2 + 1j*w*(r1*c1 + r2*c2 + r1*c2) + 1
        ret.append(np.abs(num/denom))
    return 20*np.log10(np.array(ret))

def inverting_amp(params, length):
    r1, r2 = params[0], params[1]
    return 20*np.log10(r2/r1 * np.ones(length))

def non_inverting_amp(params, length):
    r1, r2 = params[0], params[1]
    return 20*np.log10((1 + r2/r1) * np.ones(length))

def rc_highpass(params, length):
    r, c = params[0], params[1]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
        num = 1j * w * r * c
        denom = 1 + 1j * w * r * c
        ret.append(np.abs(num/denom))
    return 20*np.log10(np.array(ret))

def rc_lowpass(params, length):
    r, c = params[0], params[1]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
        num = 1
        denom = 1 + 1j * w * r * c
        ret.append(np.abs(num/denom))
    return 20*np.log10(np.array(ret))

def sk_lowpass(params, length):
    # r1, r2, c1, c2 = params[0], params[1], params[2], params[3]
    # x = np.logspace(1, 6, length)
    # ret = []
    # for w in x:
    #     num = 1
    #     denom = -1 * w * w * r1 * r2 * c1 * c2 + 1j * w * (r1 * c2 + r2*c2) + 1
    #     ret.append(np.abs(num/denom))
    # return 20*np.log10(np.array(ret))

    w0, Q = params[0], params[1]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
         num = 1
         denom = -1*w*w + w0/Q*1j*w + w0*w0
         ret.append(np.abs(num/denom))
    
    return 20*np.log10(np.array(ret))

def sk_highpass(params, length):
    # r1, r2, c1, c2 = params[0], params[1], params[2], params[3]
    # x = np.logspace(1, 6, length)
    # ret = []
    # for w in x:
    #     num = -1 * w * w
    #     denom = -1 * w * w + 1j * w * (1/(r1 * c1) + 1/(r2 * c1)) + 1/(r1 * r2 * c1 * c2)
    #     ret.append(np.abs(num/denom))
    
    # return 20*np.log10(np.array(ret))

    w0, Q = params[0], params[1]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
         num = -1 * w * w
         denom = -1*w*w + w0/Q*1j*w + w0*w0
         ret.append(np.abs(num/denom))
    
    return 20*np.log10(np.array(ret))


def voltage_divider(params, length):
    r1, r2 = params[0], params[1]
    return 20*np.log10(r2/(r1 + r2) * np.ones(length))

######################################

# --- Section 2: Create the Fixed Function Library ---

def generate_signal(name, params, length):
    """A helper to generate a signal wave."""
    if name == "SINE":
        amp, freq, phase = params
        x = np.linspace(0, 2 * np.pi, length)
        return amp * np.sin(freq * x + phase)
    elif name == "LINEAR":
        slope, intercept = params
        return slope * np.arange(length) + intercept
    elif name == "bandpass":
        return bandpass(params, length)
    elif name == "inverting_amp":
        return inverting_amp(params, length)
    elif name == "non_inverting_amp":
        return non_inverting_amp(params, length)
    elif name == "rc_highpass":
        return rc_highpass(params, length)
    elif name== "rc_lowpass":
        return rc_lowpass(params, length)
    elif name == "sk_highpass":
        return sk_highpass(params, length)
    elif name == "sk_lowpass":
        return sk_lowpass(params, length)
    elif name == "voltage_divider":
        return voltage_divider(params, length)
    return np.zeros(length)

# Create a library of fixed, non-parametric functions.
# The model will learn to choose from these specific signals.
# FIXED_FUNCTION_LIBRARY = {
#     0: torch.tensor(generate_signal("SINE", [1.0, 2.0, 0.5], SIGNAL_LENGTH), dtype=torch.float32),
#     1: torch.tensor(generate_signal("SINE", [0.7, 4.0, 1.5], SIGNAL_LENGTH), dtype=torch.float32),
#     2: torch.tensor(generate_signal("LINEAR", [0.005, 0.2], SIGNAL_LENGTH), dtype=torch.float32),
#     3: torch.tensor(generate_signal("LINEAR", [-0.003, -0.1], SIGNAL_LENGTH), dtype=torch.float32),
#     # Add more fixed, pre-defined functions here...
# }

FIXED_FUNCTION_LIBRARY = {
    #0 : torch.tensor(generate_signal("bandpass", [5000, 47e-9, 10000, 10e-9], SIGNAL_LENGTH), dtype=torch.float32),
    0 : torch.tensor(generate_signal("inverting_amp", [1, 2], SIGNAL_LENGTH), dtype=torch.float32),
    # 1 : torch.tensor(generate_signal("sk_highpass", [10000, 18000, 47e-9, 36e-9], SIGNAL_LENGTH), dtype=torch.float32),
    # 2 : torch.tensor(generate_signal("sk_lowpass", [10000, 18000, 47e-9, 36e-9], SIGNAL_LENGTH), dtype=torch.float32),
    1 : torch.tensor(generate_signal("sk_highpass", [10000, 0.707], SIGNAL_LENGTH), dtype=torch.float32),
    2 : torch.tensor(generate_signal("sk_lowpass", [10000, 0.707], SIGNAL_LENGTH), dtype=torch.float32),
}
STOP_TOKEN_INDEX = len(FIXED_FUNCTION_LIBRARY)
NUM_CLASSES = len(FIXED_FUNCTION_LIBRARY) + 1 # Add 1 for the STOP token

# --- Section 3: Model Architecture (Classification-Only) ---

class PositionalEncoding(nn.Module): # (Same as before)
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer('pe', torch.zeros(1, max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FunctionSelectorTransformer(nn.Module):
    """
    A generative model that only selects functions from a fixed library.
    It has NO regression head.
    """
    def __init__(self, signal_length: int, patch_size: int, num_classes: int,
                 d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size."
        num_patches = signal_length // patch_size
        
        self.patch_embedding = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_patches)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Only has one output head for classification
        self.classification_head = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = src.unsqueeze(1)
        src_patched = self.patch_embedding(src)
        src_permuted = src_patched.permute(0, 2, 1)
        src_pos_encoded = self.pos_encoder(src_permuted)
        encoded_patches = self.transformer_encoder(src_pos_encoded)
        aggregated_output = encoded_patches.mean(dim=1)
        classification_logits = self.classification_head(aggregated_output)
        return classification_logits

# --- Section 4: Dataset (Simplified) ---

curr_epoch = 5
idx_count = 3

class DecompositionDataset(Dataset):
    """
    Generates signals by combining functions from the FIXED library.
    The ground truth is just a sequence of class indices.
    """
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples

    # --- FIX IS HERE ---
    # The DataLoader requires this method to know the total size of the dataset.
    def __len__(self):
        return self.num_samples
    # --- END FIX ---

    # def __getitem__(self, idx):
    #     # 1. Create a ground-truth decomposition sequence of indices
    #     # Let's combine two random functions from the library
    #     idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
    #     idx2 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
        
    #     # The ground truth is now just a list of integers, ending with the STOP token
    #     true_decomposition_indices = [idx1, idx2, STOP_TOKEN_INDEX]
        
    #     # 2. Generate the signal by summing the fixed functions
    #     signal1 = FIXED_FUNCTION_LIBRARY[idx1]
    #     signal2 = FIXED_FUNCTION_LIBRARY[idx2]
    #     final_signal = signal1 + signal2 

    #     temp = sorted([idx1, idx2], key=lambda x:np.linalg.norm(FIXED_FUNCTION_LIBRARY[x] - signal1+ signal2))
    #     temp.append(STOP_TOKEN_INDEX)
        
    #     return final_signal, torch.tensor(temp, dtype=torch.long)

    def __getitem__(self, idx, check=False):
        global epoch
        if epoch < 2:
            idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            temp = [idx1, STOP_TOKEN_INDEX]

            signal = FIXED_FUNCTION_LIBRARY[idx1]

            return signal, torch.tensor(temp, dtype=torch.long)
        elif epoch < 14:
            idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            idx2 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))

            temp = [idx1, STOP_TOKEN_INDEX]

            signal = FIXED_FUNCTION_LIBRARY[idx1]
            distractor_amplitude = np.random.uniform(0.1, epoch/20)
            signal = signal + distractor_amplitude * FIXED_FUNCTION_LIBRARY[idx2]

            return signal, torch.tensor(temp, dtype=torch.long)
        else:
            # idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            # idx2 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            
            # # The ground truth is now just a list of integers, ending with the STOP token
            # true_decomposition_indices = [idx1, idx2, STOP_TOKEN_INDEX]
            
            # # 2. Generate the signal by summing the fixed functions
            # signal1 = FIXED_FUNCTION_LIBRARY[idx1]
            # signal2 = FIXED_FUNCTION_LIBRARY[idx2]
            # final_signal = signal1 + signal2 

            # temp = sorted([idx1, idx2], key=lambda x:np.linalg.norm(FIXED_FUNCTION_LIBRARY[x] - signal1 - signal2))
            # temp.append(STOP_TOKEN_INDEX)
            
            # return final_signal, torch.tensor(temp, dtype=torch.long)
            # idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            # idx2 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
            # idx3 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))

            # true_decomposition_indices = [idx1, idx2, idx3,STOP_TOKEN_INDEX]

            # signal1 = FIXED_FUNCTION_LIBRARY[idx1]
            # signal2 = FIXED_FUNCTION_LIBRARY[idx2]
            # signal3 = FIXED_FUNCTION_LIBRARY[idx3]

            # final_signal = signal1 + signal2 + signal3
            # temp = sorted([idx1, idx2, idx3], key=lambda x:np.linalg.norm(FIXED_FUNCTION_LIBRARY[x] - signal1 - signal2 - signal3 ))
            # temp.append(STOP_TOKEN_INDEX)

            # return final_signal, torch.tensor(temp, dtype=torch.long)

            # FIXED_FUNCTION_LIBRARY = {
            #     0 : torch.tensor(generate_signal("bandpass", [5000, 47e-9, 10000, 10e-9], SIGNAL_LENGTH), dtype=torch.float32),
            #     1 : torch.tensor(generate_signal("inverting_amp", [1, 2], SIGNAL_LENGTH), dtype=torch.float32),
            #     2 : torch.tensor(generate_signal("sk_highpass", [10000, 18000, 47e-9, 36e-9], SIGNAL_LENGTH), dtype=torch.float32),
            #     3 : torch.tensor(generate_signal("sk_lowpass", [10000, 18000, 47e-9, 36e-9], SIGNAL_LENGTH), dtype=torch.float32),
            # }
            global curr_epoch
            global idx_count
            import random
            if not curr_epoch == epoch:
                #idx_count = np.random.randint(0, int(len(FIXED_FUNCTION_LIBRARY)*1.5))
                idx_count = 1+epoch//6
                curr_epoch = epoch
            temp = random.choices(list(FIXED_FUNCTION_LIBRARY.keys()), k=idx_count)

            common_resistor_values_ohms = [10, 22, 33, 47, 68, 100, 220, 330, 470, 680, 1000, 2200, 3300, 4700, 6800, 10000, 22000, 33000, 47000, 68000, 100000, 220000, 330000, 470000, 680000, 1000000]
            common_capacitor_values_farads = [10e-12, 22e-12, 33e-12, 47e-12, 68e-12, 100e-12, 220e-12, 330e-12, 470e-12, 680e-12, 1e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9, 220e-9, 330e-9, 470e-9, 680e-9, 1e-6, 2.2e-6, 3.3e-6, 4.7e-6, 10e-6, 22e-6, 33e-6, 47e-6, 100e-6, 220e-6, 330e-6, 470e-6, 1000e-6]
            common_cutoff_values = np.logspace(1, 6, 10) # 10 values from 10Hz to 1MHz
            common_q_values = [0.5, 0.707, 1.0, 1.414, 2.0] # Common Q values for filters
            all_params = []

            signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32)
            for idx in temp:
                # if idx == 0:
                #     params = [random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads), random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads)]
                #     signal = signal + torch.tensor(generate_signal("bandpass", params, SIGNAL_LENGTH), dtype=torch.float32)
                if idx == 0:
                    params = [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)]
                    signal = signal + torch.tensor(generate_signal("inverting_amp", params, SIGNAL_LENGTH), dtype=torch.float32)
                if idx == 1:
                    #params = [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads), random.choice(common_capacitor_values_farads)]
                    params = [random.choice(common_cutoff_values), random.choice(common_q_values)]
                    signal = signal + torch.tensor(generate_signal("sk_highpass", params, SIGNAL_LENGTH), dtype=torch.float32)
                if idx == 2:
                    #params = [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads), random.choice(common_capacitor_values_farads)]
                    params = [random.choice(common_cutoff_values), random.choice(common_q_values)]
                    signal = signal + torch.tensor(generate_signal("sk_lowpass", params, SIGNAL_LENGTH), dtype=torch.float32)
                all_params.append(params)
                #signal = signal + FIXED_FUNCTION_LIBRARY[idx]
            

            temp = sorted(temp, key = lambda x : np.linalg.norm(FIXED_FUNCTION_LIBRARY[x] - signal))
            temp.append(STOP_TOKEN_INDEX)
            if check:
                return signal, torch.tensor(temp, dtype=torch.long), all_params
            return signal, torch.tensor(temp, dtype=torch.long)

        # else:
        #     idx1 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
        #     idx2 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))
        #     idx3 = np.random.randint(0, len(FIXED_FUNCTION_LIBRARY))

        #     true_decomposition_indices = [idx1, idx2, idx3,STOP_TOKEN_INDEX]

        #     signal1 = FIXED_FUNCTION_LIBRARY[idx1]
        #     signal2 = FIXED_FUNCTION_LIBRARY[idx2]
        #     signal3 = FIXED_FUNCTION_LIBRARY[idx3]

        #     final_signal = signal1 + signal2 + signal3
        #     temp = sorted([idx1, idx2, idx3], key=lambda x:np.linalg.norm(FIXED_FUNCTION_LIBRARY[x] - signal1 - signal2 - signal3 ))
        #     temp.append(STOP_TOKEN_INDEX)

        #     return final_signal, torch.tensor(temp, dtype=torch.long)





    # def __getitem__(self, idx):
    #     import random
    #     func_num = random.randint(1, len(FIXED_FUNCTION_LIBRARY))




def evaluate_epoch(model, validation_dataloader, loss_fn, device):
    """
    Correctly evaluates the model on the validation dataset.
    This version ensures it ONLY uses the model's own predictions for reconstruction.
    """
    print("\n--- Running Validation Epoch ---")
    model.eval() # Set the model to evaluation mode
    total_main_loss = 0
    total_recon_loss = 0
    
    # We only need the CrossEntropyLoss for the main loss calculation here
    class_loss_fn = loss_fn

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch_idx, (initial_signals, true_indices_seqs) in enumerate(validation_dataloader):
            initial_signals = initial_signals.to(device)
            true_indices_seqs = true_indices_seqs.to(device)

            # --- Perform autoregressive prediction for the entire batch ---
            current_residuals = initial_signals.clone()
            
            # This list will store the model's ACTUAL predictions at each step.
            batch_predictions_indices = [[] for _ in range(initial_signals.size(0))]
            
            main_loss_for_batch = torch.tensor(0.0, device=device)
            max_seq_len = true_indices_seqs.size(1)

            for step_idx in range(max_seq_len):
                pred_logits = model(current_residuals)
                true_labels_for_step = true_indices_seqs[:, step_idx]
                
                # Calculate the main classification loss against the ground truth
                loss = class_loss_fn(pred_logits, true_labels_for_step)
                main_loss_for_batch = main_loss_for_batch + loss.mean()

                # Get the model's OWN prediction for this step
                predicted_indices_for_step = torch.argmax(pred_logits, dim=1)
                
                # Store the prediction
                for i in range(len(predicted_indices_for_step)):
                    batch_predictions_indices[i].append(predicted_indices_for_step[i].item())
                
                # Update the residual using the model's own prediction
                signals_to_subtract = torch.stack([
                    FIXED_FUNCTION_LIBRARY.get(idx.item(), torch.zeros(SIGNAL_LENGTH))
                    for idx in predicted_indices_for_step
                ]).to(device)
                current_residuals = current_residuals - signals_to_subtract
            
            # --- Calculate Reconstruction Loss using the stored predictions ---
            recon_loss_for_batch = torch.tensor(0.0, device=device)
            for i in range(initial_signals.size(0)):
                reconstructed_signal = torch.zeros(SIGNAL_LENGTH, device=device)
                # This loop now correctly uses the predictions we stored
                for class_idx in batch_predictions_indices[i]:
                    if class_idx != STOP_TOKEN_INDEX:
                        reconstructed_signal += FIXED_FUNCTION_LIBRARY[class_idx].to(device)
                
                # Use a simple MSE for reconstruction loss
                recon_loss_for_batch += nn.functional.mse_loss(reconstructed_signal, initial_signals[i])
            
            total_main_loss += (main_loss_for_batch / max_seq_len).item()
            total_recon_loss += (recon_loss_for_batch / len(initial_signals)).item()

            # --- Add a debug print for the first item in the first batch of each epoch ---
            if batch_idx == 0:
                print("--- Validation Debug Print (Epoch First Batch) ---")
                print(f"Sample 0 Ground Truth Indices: {true_indices_seqs[0].cpu().numpy()}")
                print(f"Sample 0 Predicted Indices:   {batch_predictions_indices[0]}")
                print("-" * 50)
    model.eval()
    with torch.no_grad():
        ds = DecompositionDataset()
        predicted_index = -1
        initial_signals, true_indices_seqs = ds.__getitem__(0)
        temp = initial_signals
        total = torch.zeros(initial_signals.numel())
        res = []
        while not predicted_index == 3:
            input_tensor = initial_signals.unsqueeze(0)
            input_tensor = input_tensor.to(device)
            output_logits = model(input_tensor)
            predicted_index = torch.argmax(output_logits, dim=1).item()
            res.append(predicted_index)
            if predicted_index == 3:
                break
            initial_signals = initial_signals - FIXED_FUNCTION_LIBRARY[predicted_index]
            total = total + FIXED_FUNCTION_LIBRARY[predicted_index]
        
    # import matplotlib.pyplot as plt

    # print("***************")
    # print("graph demonstration:")
    # print(f"actual: {true_indices_seqs}")
    # print(f"predicted: {res}")
    # print("***************")

    # f = np.logspace(1, 6, initial_signals.numel())

    # plt.figure()
    # plt.semilogx(f, temp.numpy())
    # plt.semilogx(f, total.numpy())
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Magnitude (dB)")
    # plt.title("Sallen‑Key Low‑Pass Filter – Magnitude Response")
    # plt.grid(True, which='both')
    # plt.show()

        




    avg_main_loss = total_main_loss / len(validation_dataloader)
    avg_recon_loss = total_recon_loss / len(validation_dataloader)
    
    return avg_main_loss, avg_recon_loss

# --- Section 5: The Simplified Training Function ---

# Add a new hyperparameter at the top of your script
RECONSTRUCTION_LOSS_WEIGHT = 0.1

def train_epoch(model, dataloader, optimizer, loss_fns, device, weights):
    model.train()
    total_epoch_loss = 0
    class_loss_fn, recon_loss_fn = loss_fns # Loss fns will now be (CrossEntropy, MSE)
    stop_weight, recon_weight = weights

    for initial_signals, true_indices_seqs in dataloader:
        initial_signals = initial_signals.to(device)
        true_indices_seqs = true_indices_seqs.to(device)
        optimizer.zero_grad()
        
        batch_loss = torch.tensor(0.0, device=device)
        
        # We need to store predictions for the reconstruction loss
        batch_predictions = [[] for _ in range(initial_signals.size(0))]

        current_residuals = initial_signals.clone()
        max_seq_len = true_indices_seqs.size(1)

        for step_idx in range(max_seq_len):
            pred_logits = model(current_residuals)
            true_labels_for_step = true_indices_seqs[:, step_idx]
            
            # Store the model's actual prediction for this step
            predicted_indices = torch.argmax(pred_logits, dim=1)
            for i in range(len(predicted_indices)):
                batch_predictions[i].append(predicted_indices[i].item())

            # --- Main decomposition loss (teacher-forced) ---
            loss = class_loss_fn(pred_logits, true_labels_for_step)
            is_stop_mask = (true_labels_for_step == STOP_TOKEN_INDEX)
            loss_weights = torch.where(is_stop_mask, stop_weight, 1.0)
            batch_loss = batch_loss + (loss * loss_weights).mean()

            # --- Update residual with ground truth for next step ---
            signals_to_subtract = torch.stack([
                FIXED_FUNCTION_LIBRARY.get(idx.item(), torch.zeros(SIGNAL_LENGTH)) 
                for idx in true_labels_for_step
            ]).to(device)
            current_residuals = current_residuals - signals_to_subtract

        # --- NEW: Calculate Reconstruction Loss ---
        reconstruction_loss = torch.tensor(0.0, device=device)
        for i in range(initial_signals.size(0)):
            reconstructed_signal = torch.zeros(SIGNAL_LENGTH, device=device)
            for class_idx in batch_predictions[i]:
                if class_idx != STOP_TOKEN_INDEX:
                    reconstructed_signal += FIXED_FUNCTION_LIBRARY[class_idx].to(device)

            # Add the MSE between original and reconstructed signal
            reconstruction_loss += recon_loss_fn(reconstructed_signal, initial_signals[i])
        
        # Add the weighted reconstruction loss to the main loss
        batch_loss = batch_loss + (recon_weight * (reconstruction_loss / len(initial_signals)))

        normalized_batch_loss = batch_loss / max_seq_len
        normalized_batch_loss.backward()
        optimizer.step()
        
        total_epoch_loss += normalized_batch_loss.item()
        
    return total_epoch_loss / len(dataloader)

# You would also need to update your main block to pass the correct loss functions
# loss_fns = (nn.CrossEntropyLoss(reduction='none'), nn.MSELoss())
# weights = (STOP_LOSS_WEIGHT, RECONSTRUCTION_LOSS_WEIGHT)

# --- Section 6: Main Execution Block ---

SAVE_START_EPOCH = 18
MODELS_DIR = "./saved_models"

if __name__ == '__main__':
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import random_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Create the full dataset
    full_dataset = DecompositionDataset(num_samples=8000)

    # 2. Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset)) # 80% for training
    val_size = len(full_dataset) - train_size # 20% for validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. Create separate DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE) # No need to shuffle validation data

    # 4. Model
    model = FunctionSelectorTransformer(
        signal_length=SIGNAL_LENGTH, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES,
        d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=N_LAYERS
    ).to(device)

    # Define both loss functions
    loss_fns = (nn.CrossEntropyLoss(reduction='none'), nn.MSELoss())
    # Define both weights
    weights = (STOP_LOSS_WEIGHT, RECONSTRUCTION_LOSS_WEIGHT)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        # The train_epoch function also needs to return both losses if you want to monitor them
        # For simplicity, we'll just show the call to the updated evaluate_epoch
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fns, device, weights)
        
        # --- UPDATED CALL ---
        # Call the new evaluation function, which returns two values
        # The evaluation call is now simpler
        avg_val_loss, avg_val_recon_loss = evaluate_epoch(model, val_dataloader, loss_fn, device)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Recon Loss: {avg_val_recon_loss:.4f}")

        if (epoch + 1) >= SAVE_START_EPOCH:
            # Format validation loss for filename (e.g., 0.2345 -> 0_2345)
            # Use a slightly more robust formatting for decimal points
            formatted_val_loss = f"{avg_val_loss:.4f}".replace('.', '_')
            model_filename = f"model_epoch{epoch+1}_val_loss_{formatted_val_loss}.pth"
            save_path = os.path.join(MODELS_DIR, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    print("--- Finished Training ---")