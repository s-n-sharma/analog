import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Section 1: Constants and Configuration ---

# Signal and Model Hyperparameters
SIGNAL_LENGTH = 400
PATCH_SIZE = 20
D_MODEL = 256
N_HEAD = 16
N_LAYERS = 8

# Training Hyperparameters
EPOCHS = 100
LR = 0.0001
BATCH_SIZE = 32

# Loss Weight for the STOP token
STOP_LOSS_WEIGHT = 1.2
# Reconstruction Loss Weight: Critical for training the classifier
# to yield outputs that lead to good reconstruction via the optimizer.
RECONSTRUCTION_LOSS_WEIGHT = 1.0

# Model Saving Configuration
MODELS_DIR = "saved_models"
SAVE_START_EPOCH = 10 # Start saving models after this epoch
SAVE_EVERY_N_EPOCHS = 5 # Save a model checkpoint every N epochs

# --- Optimizer Configuration for Validation/Inference and Training Recon Loss ---
OPTIMIZER_METHOD = 'L-BFGS-B' # A good general-purpose method for bounded problems
MAX_OPTIMIZER_ITERATIONS = 200 # Increased iterations for better fit
TOLERANCE = 1e-7 # Tighter tolerance for optimization convergence
# --------------------------------------------------------------------------


# --- Section 2: Signal Generation Functions and Parameter Ranges ---

# Base signal generation functions
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
    w0, Q = params[0], params[1]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
         num = 1
         denom = -1*w*w + w0/Q*1j*w + w0*w0
         ret.append(np.abs(num/denom))
    return 20*np.log10(np.array(ret))

def sk_highpass(params, length):
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

# Helper wrapper to call specific generation functions by name
def _generate_signal_func_wrapper(name, params, length):
    """A wrapper for specific signal generation functions."""
    if name == "SINE": # Example for future expansion
        amp, freq, phase = params
        x = np.linspace(0, 2 * np.pi, length)
        return amp * np.sin(freq * x + phase)
    elif name == "LINEAR": # Example for future expansion
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

# --- Parameter ranges for each function type (tuned for realistic values) ---
common_resistor_values_ohms = [10, 22, 33, 47, 68, 100, 220, 330, 470, 680, 1000, 2200, 3300, 4700, 6800, 10000, 22000, 33000, 47000, 68000, 100000, 220000, 330000, 470000, 680000, 1000000]
common_capacitor_values_farads = [10e-12, 22e-12, 33e-12, 47e-12, 68e-12, 100e-12, 220e-12, 330e-12, 470e-12, 680e-12, 1e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9, 220e-9, 330e-9, 470e-9, 680e-9, 1e-6, 2.2e-6, 3.3e-6, 4.7e-6, 10e-6, 22e-6, 33e-6, 47e-6, 100e-6, 220e-6, 330e-6, 470e-6, 1000e-6]
common_cutoff_values = np.logspace(1, 6, 100) # More values for optimization, from 10Hz to 1MHz
common_q_values = [0.5, 0.707, 1.0, 1.414, 2.0, 3.0, 5.0, 10.0] # More Q values

# Map function names to their actual generation functions and parameter bounds
FUNCTION_SPECS = {
    "inverting_amp": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "inverting_amp",
        "num_params": 2,
        "bounds": [(10, 1e6), (10, 1e6)], # R1, R2 ranges
        "initial_guess_strategy": lambda: [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)]
    },
    "sk_highpass": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "sk_highpass",
        "num_params": 2,
        "bounds": [(10, 1e6), (0.1, 10.0)], # w0 (cutoff freq), Q (damping)
        "initial_guess_strategy": lambda: [random.choice(common_cutoff_values), random.choice(common_q_values)]
    },
    "sk_lowpass": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "sk_lowpass",
        "num_params": 2,
        "bounds": [(10, 1e6), (0.1, 10.0)], # w0 (cutoff freq), Q (damping)
        "initial_guess_strategy": lambda: [random.choice(common_cutoff_values), random.choice(common_q_values)]
    },
    "voltage_divider": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "voltage_divider",
        "num_params": 2,
        "bounds": [(10, 1e6), (10, 1e6)], # R1, R2 ranges
        "initial_guess_strategy": lambda: [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)]
    },
    "rc_highpass": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "rc_highpass",
        "num_params": 2,
        "bounds": [(10, 1e6), (10e-12, 1000e-6)], # R, C ranges
        "initial_guess_strategy": lambda: [random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads)]
    },
    "rc_lowpass": {
        "func": _generate_signal_func_wrapper,
        "params_gen_name": "rc_lowpass",
        "num_params": 2,
        "bounds": [(10, 1e6), (10e-12, 1000e-6)], # R, C ranges
        "initial_guess_strategy": lambda: [random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads)]
    },
}

# Define FIXED_FUNCTION_LIBRARY_INFO as a lookup for names, not pre-computed signals
# The order here defines the integer indices (0, 1, 2, ...) that the classifier predicts
FIXED_FUNCTION_LIBRARY_INFO = {
    0: "inverting_amp",
    1: "sk_highpass",
    2: "sk_lowpass",
    3: "voltage_divider",
    4: "rc_highpass",
    5: "rc_lowpass",
}
STOP_TOKEN_INDEX = len(FIXED_FUNCTION_LIBRARY_INFO)
NUM_CLASSES = len(FIXED_FUNCTION_LIBRARY_INFO) + 1 # Add 1 for the STOP token


# --- Section 3: Model Architecture ---

class PositionalEncoding(nn.Module):
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
    A generative model that selects function types from FIXED_FUNCTION_LIBRARY_INFO.
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

# --- Section 4: Dataset ---
# Global epoch variable for dataset generation logic
# This will be updated by the main training loop
#global epoch
epoch = 0 # Initial value for the global epoch


class DecompositionDataset(Dataset):
    """
    Generates signals by combining functions from the FIXED_FUNCTION_LIBRARY_INFO
    with variable parameters, progressively increasing complexity based on global epoch.
    The ground truth is a sequence of class indices and their true parameters.
    """
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Access the global epoch variable to control complexity
        global epoch

        num_function_types = len(FIXED_FUNCTION_LIBRARY_INFO)

        # Progressive increase in number of components based on epoch
        if epoch < 5:
            num_components_to_sum = 1
        elif epoch < 20:
            num_components_to_sum = np.random.randint(1, 3) # 1 or 2 components
        elif epoch < 40:
            num_components_to_sum = np.random.randint(1, min(4, num_function_types + 1)) # 1 to 3 components (or more if more types added)
        elif epoch < 60:
            num_components_to_sum = np.random.randint(1, min(5, num_function_types + 1)) # 1 to 4 components (or more)
        elif epoch < 80: # Added a new stage for 5 components
            num_components_to_sum = np.random.randint(1, min(6, num_function_types + 1)) # 1 to 5 components (or more)
        else: # For epochs >= 80, potentially up to max available functions or a higher cap
            num_components_to_sum = np.random.randint(1, min(7, num_function_types + 1)) # 1 to 6 components (or more)


        # Randomly choose function types (allowing repetition)
        # Using random.choices is appropriate if you want to allow picking the same function multiple times
        chosen_base_indices = random.choices(list(FIXED_FUNCTION_LIBRARY_INFO.keys()), k=num_components_to_sum)
        
        signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32)
        true_decomposition_indices = []
        true_decomposition_params = [] # To store the actual parameters used for generation

        for func_idx in chosen_base_indices:
            func_name = FIXED_FUNCTION_LIBRARY_INFO[func_idx]
            func_spec = FUNCTION_SPECS[func_name]
            
            # Generate random parameters within the defined bounds for this function type
            params = []
            for i, (lower_bound, upper_bound) in enumerate(func_spec["bounds"]):
                # Handle log-uniform sampling for frequency-like parameters
                if func_name in ["sk_highpass", "sk_lowpass", "rc_highpass", "rc_lowpass"] and i == 0: # Assuming w0 or R is the first param
                     param_val = np.exp(np.random.uniform(np.log(lower_bound), np.log(upper_bound)))
                elif func_name in ["rc_highpass", "rc_lowpass"] and i == 1: # For C (capacitor)
                     param_val = np.exp(np.random.uniform(np.log(lower_bound), np.log(upper_bound)))
                else: # Linear uniform sampling for other parameters (e.g., resistances, Q values)
                    param_val = np.random.uniform(lower_bound, upper_bound)
                params.append(param_val)

            # Generate the signal component using the *randomly chosen parameters*
            component_signal = torch.tensor(
                func_spec["func"](func_spec["params_gen_name"], params, SIGNAL_LENGTH), 
                dtype=torch.float32
            )
            signal += component_signal

            true_decomposition_indices.append(func_idx)
            true_decomposition_params.append(params) # Store the actual parameters

        # Sort the ground truth indices and their corresponding parameters for consistent ground truth sequence
        # This is important for the classification loss where order matters.
        # Sorting by the function index (the first element of the zipped pair)
        combined_truth = sorted(zip(true_decomposition_indices, true_decomposition_params), key=lambda x: x[0])
        true_decomposition_indices = [item[0] for item in combined_truth]
        true_decomposition_params = [item[1] for item in combined_truth]
        
        # Append STOP token and its placeholder params
        true_decomposition_indices.append(STOP_TOKEN_INDEX)
        true_decomposition_params.append(None) # No parameters for STOP token

        # Optional: Add a small amount of Gaussian noise to the final signal
        signal += torch.randn(SIGNAL_LENGTH) * 0.1 # Adjust noise amplitude as needed, e.g., 0.05 to 0.2

        # ALWAYS return 3 values (signal, indices, params)
        return signal, torch.tensor(true_decomposition_indices, dtype=torch.long), true_decomposition_params


# --- Section 5: Optimization Objective Function (for Validation/Inference and Training Recon Loss) ---

def objective_function(params, target_signal_np, function_name_key):
    """
    Objective function for scipy.optimize.minimize.
    Calculates MSE between the target_signal and a signal generated with given parameters.
    """
    func_spec = FUNCTION_SPECS[function_name_key]
    generated_signal = func_spec["func"](func_spec["params_gen_name"], params.tolist(), SIGNAL_LENGTH)
    mse = np.mean((generated_signal - target_signal_np)**2)
    return mse

# --- Section 6: Training and Evaluation Functions ---

def train_epoch(model, dataloader, optimizer, loss_fns, device, weights):
    model.train()
    total_epoch_loss = 0
    class_loss_fn, recon_loss_fn = loss_fns # CrossEntropy, MSE
    stop_weight, recon_weight = weights

    for initial_signals, true_indices_seqs, true_params_seqs in dataloader: # Now unpacks 3 items
        initial_signals = initial_signals.to(device)
        true_indices_seqs = true_indices_seqs.to(device)
        # true_params_seqs are lists of lists, no need to move to device directly for this part
        
        optimizer.zero_grad()
        
        batch_classification_loss = torch.tensor(0.0, device=device)
        
        # Store model's predictions for reconstruction loss calculation
        batch_predictions_for_recon = [[] for _ in range(initial_signals.size(0))]

        current_residuals_classifier = initial_signals.clone() # Residual for classifier's input
        max_seq_len = true_indices_seqs.size(1)

        # Teacher-forcing loop for classification training
        for step_idx in range(max_seq_len):
            pred_logits = model(current_residuals_classifier)
            true_labels_for_step = true_indices_seqs[:, step_idx]
            
            # Get the model's actual prediction for this step
            predicted_indices_by_model = torch.argmax(pred_logits, dim=1)
            for i in range(len(predicted_indices_by_model)):
                batch_predictions_for_recon[i].append(predicted_indices_by_model[i].item())

            # --- Main classification loss (teacher-forced) ---
            loss = class_loss_fn(pred_logits, true_labels_for_step)
            is_stop_mask = (true_labels_for_step == STOP_TOKEN_INDEX)
            loss_weights = torch.where(is_stop_mask, stop_weight, 1.0)
            batch_classification_loss += (loss * loss_weights).mean()

            # --- Update residual for classifier's NEXT input (teacher forcing) ---
            # This is critical for the classifier to learn to identify components
            # from the remaining signal. We use the *ground truth* to update the residual.
            signals_to_subtract_gt = torch.zeros_like(initial_signals)
            for i in range(initial_signals.size(0)):
                true_idx = true_labels_for_step[i].item()
                # Ensure we don't go out of bounds for true_params_seqs[i] or try to use None params
                if (true_idx != STOP_TOKEN_INDEX and 
                    step_idx < len(true_params_seqs[i]) and 
                    true_params_seqs[i][step_idx] is not None):
                    
                    func_name_gt = FIXED_FUNCTION_LIBRARY_INFO[true_idx]
                    func_spec_gt = FUNCTION_SPECS[func_name_gt]
                    
                    gt_params = true_params_seqs[i][step_idx] # Get ground truth parameters
                    
                    component_gt_signal_np = func_spec_gt["func"](func_spec_gt["params_gen_name"], gt_params, SIGNAL_LENGTH)
                    component_gt_signal_torch = torch.tensor(component_gt_signal_np, dtype=torch.float32).to(device)
                    signals_to_subtract_gt[i] = component_gt_signal_torch
                # else: for STOP token or if params are None, subtract nothing
            current_residuals_classifier = current_residuals_classifier - signals_to_subtract_gt


        # --- Calculate Reconstruction Loss using the model's PREDICTED components + optimizer ---
        # This part ensures the classifier learns to predict types that lead to good overall reconstruction.
        reconstruction_loss_val = torch.tensor(0.0, device=device)
        for i in range(initial_signals.size(0)): # Iterate through batch items
            current_residual_for_recon = initial_signals[i].clone().to(device)
            item_reconstructed_signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32, device=device)

            # Loop through the model's predictions for this single item
            for pred_idx in batch_predictions_for_recon[i]:
                if pred_idx == STOP_TOKEN_INDEX:
                    break # Stop token predicted, end sequence

                if pred_idx not in FIXED_FUNCTION_LIBRARY_INFO:
                    # If model predicts an invalid index, skip or penalize this.
                    # For now, we skip and let MSE handle the discrepancy.
                    continue

                function_name = FIXED_FUNCTION_LIBRARY_INFO[pred_idx]
                func_spec = FUNCTION_SPECS[function_name]

                initial_guess = func_spec["initial_guess_strategy"]()
                bounds = func_spec["bounds"]

                # Convert current_residual_for_recon to NumPy for SciPy's optimizer
                target_signal_np = current_residual_for_recon.cpu().numpy()

                try:
                    result = minimize(
                        objective_function,
                        initial_guess,
                        args=(target_signal_np, function_name),
                        method=OPTIMIZER_METHOD,
                        bounds=bounds,
                        options={'maxiter': MAX_OPTIMIZER_ITERATIONS, 'ftol': TOLERANCE}
                    )
                    optimized_params = result.x.tolist()
                    
                    optimized_component_signal_np = func_spec["func"](func_spec["params_gen_name"], optimized_params, SIGNAL_LENGTH)
                    optimized_component_signal_torch = torch.tensor(optimized_component_signal_np, dtype=torch.float32).to(device)

                    # Update residual and reconstruction for this item
                    current_residual_for_recon -= optimized_component_signal_torch
                    item_reconstructed_signal += optimized_component_signal_torch

                except Exception as e:
                    # If optimizer fails during training recon loss calc, means this path is bad.
                    # The MSE will be high, penalizing the classifier for picking this path.
                    # print(f"Warning: Optimizer failed during training recon loss calc for {func_name}: {e}")
                    pass # Continue loop; failure will contribute to high MSE

            # MSE between the item's original signal and its full reconstructed signal
            reconstruction_loss_val += recon_loss_fn(item_reconstructed_signal, initial_signals[i])
        
        # Combine losses
        # Normalize classification loss by sequence length (average over steps)
        normalized_classification_loss = batch_classification_loss / max_seq_len
        # Normalize reconstruction loss by batch size (average over samples)
        normalized_reconstruction_loss = reconstruction_loss_val / len(initial_signals)

        final_total_loss = normalized_classification_loss + (recon_weight * normalized_reconstruction_loss)
        
        final_total_loss.backward()
        optimizer.step()
        
        total_epoch_loss += final_total_loss.item()
        
    return total_epoch_loss / len(dataloader)


def evaluate_epoch(model, validation_dataloader, loss_fn, device):
    """
    Evaluates the model by running the full classifier + optimizer decomposition pipeline.
    """
    print("\n--- Running Validation Epoch ---")
    model.eval() # Set the model to evaluation mode
    total_main_classification_loss = 0
    total_reconstruction_loss = 0
    
    class_loss_fn = loss_fn # CrossEntropyLoss for classification part

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch_idx, (initial_signals, true_indices_seqs, true_params_seqs) in enumerate(validation_dataloader):
            initial_signals = initial_signals.to(device)
            true_indices_seqs = true_indices_seqs.to(device)
            # true_params_seqs are lists of lists, no need to move to device directly for this part

            # --- Perform autoregressive decomposition for the entire batch ---
            batch_current_residuals = initial_signals.clone()
            batch_reconstructed_signals = torch.zeros_like(initial_signals)
            
            # Store full prediction info for debug
            batch_predicted_decomposition_info = [[] for _ in range(initial_signals.size(0))]
            
            main_classification_loss_for_batch = torch.tensor(0.0, device=device)
            max_seq_len_gt = true_indices_seqs.size(1)

            # Max steps for autoregressive inference to prevent infinite loops
            # Allowing for more components than ground truth to see if it over-predicts
            max_inference_steps = max_seq_len_gt + 3 # Ground truth max + a few extra

            # Loop for each step of decomposition
            for step_idx in range(max_inference_steps):
                pred_logits = model(batch_current_residuals)
                predicted_indices_for_step = torch.argmax(pred_logits, dim=1)
                
                # --- Calculate Classification Loss (against ground truth, teacher-forced for this metric) ---
                if step_idx < max_seq_len_gt: # Only if we have ground truth for this step
                    true_labels_for_step = true_indices_seqs[:, step_idx]
                    loss = class_loss_fn(pred_logits, true_labels_for_step)
                    is_stop_mask = (true_labels_for_step == STOP_TOKEN_INDEX)
                    loss_weights = torch.where(is_stop_mask, STOP_LOSS_WEIGHT, 1.0)
                    main_classification_loss_for_batch += (loss * loss_weights).mean()

                # --- Update residuals and build reconstruction using predicted types + optimizer ---
                all_stopped_in_batch = True
                for i in range(initial_signals.size(0)): # Iterate through each item in the batch
                    # If this item has already predicted STOP or is somehow done, skip it
                    if batch_predicted_decomposition_info[i] and batch_predicted_decomposition_info[i][-1]['index'] == STOP_TOKEN_INDEX:
                        continue # This item is done for this batch, move to next

                    predicted_idx = predicted_indices_for_step[i].item()

                    if predicted_idx == STOP_TOKEN_INDEX:
                        batch_predicted_decomposition_info[i].append({"index": predicted_idx, "params": None})
                        continue # This item is done, move to next item in batch

                    all_stopped_in_batch = False # At least one item is still actively predicting

                    if predicted_idx not in FIXED_FUNCTION_LIBRARY_INFO:
                        # Invalid prediction, record it and potentially stop processing this item further
                        batch_predicted_decomposition_info[i].append({"index": predicted_idx, "params": None})
                        continue # Skip processing this invalid prediction

                    function_name = FIXED_FUNCTION_LIBRARY_INFO[predicted_idx]
                    func_spec = FUNCTION_SPECS[function_name]

                    initial_guess = func_spec["initial_guess_strategy"]()
                    bounds = func_spec["bounds"]

                    target_signal_np = batch_current_residuals[i].cpu().numpy()

                    try:
                        result = minimize(
                            objective_function,
                            initial_guess,
                            args=(target_signal_np, function_name),
                            method=OPTIMIZER_METHOD,
                            bounds=bounds,
                            options={'maxiter': MAX_OPTIMIZER_ITERATIONS, 'ftol': TOLERANCE}
                        )
                        optimized_params = result.x.tolist()
                        
                        optimized_component_signal_np = func_spec["func"](func_spec["params_gen_name"], optimized_params, SIGNAL_LENGTH)
                        optimized_component_signal_torch = torch.tensor(optimized_component_signal_np, dtype=torch.float32).to(device)

                        # Update residual and reconstruction for this item
                        batch_current_residuals[i] -= optimized_component_signal_torch
                        batch_reconstructed_signals[i] += optimized_component_signal_torch
                        
                        batch_predicted_decomposition_info[i].append({"index": predicted_idx, "params": optimized_params})

                    except Exception as e:
                        # If optimizer fails for this specific item, record the prediction with None params
                        # and mark it as potentially 'stuck' or 'finished' for this item
                        # print(f"   Validation Optimizer failed for {function_name} on item {i}: {e}")
                        batch_predicted_decomposition_info[i].append({"index": predicted_idx, "params": None}) 
                        # This item will effectively stop processing for this step if optimization failed.
                        continue # Continue to next item in batch or next step

                if all_stopped_in_batch:
                    break # All items in batch have predicted STOP or are stuck

            # --- Calculate overall Reconstruction Loss for the batch ---
            total_reconstruction_loss_for_batch = nn.functional.mse_loss(batch_reconstructed_signals, initial_signals, reduction='sum')

            # Accumulate overall batch losses (normalize by number of elements)
            if max_seq_len_gt > 0:
                total_main_classification_loss += (main_classification_loss_for_batch / max_seq_len_gt).item()
            else:
                total_main_classification_loss += main_classification_loss_for_batch.item() # If sequence length is 0 (empty GT)

            total_reconstruction_loss += (total_reconstruction_loss_for_batch / initial_signals.size(0)).item()

            # --- Debug print for the first item in the first batch ---
            if batch_idx == 0:
                print("--- Validation Debug Print (Epoch First Batch) ---")
                print(f"Sample 0 Ground Truth Indices: {true_indices_seqs[0].cpu().numpy()}")
                # Display only the indices part for predicted sequence for brevity
                predicted_indices_debug = [p['index'] for p in batch_predicted_decomposition_info[0]]
                print(f"Sample 0 Predicted Indices (Classifier): {predicted_indices_debug}")
                print(f"Sample 0 Ground Truth Params: {true_params_seqs[0]}")
                # Display only actual optimized params for brevity
                predicted_params_debug = [p['params'] for p in batch_predicted_decomposition_info[0] if p['params'] is not None]
                print(f"Sample 0 Predicted Params (Optimizer): {predicted_params_debug}")
                print("-" * 50)
    
    # --- Plotting a single example ---
    # Use a consistent example from the validation set (e.g., the very first one)
    # Ensure this is independent of batch_size/dataloader shuffling for consistent plots.
    model.eval()
    with torch.no_grad():
        # Access a single sample directly from the validation dataset (not dataloader batch)
        # This provides a consistent example for plotting across epochs.
        single_val_signal, single_val_true_indices, single_val_true_params = validation_dataloader.dataset[0]
        single_val_signal = single_val_signal.to(device)
        
        plot_residual = single_val_signal.clone()
        plot_reconstructed_signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32, device=device)
        plot_predicted_indices = []
        plot_predicted_params = []

        max_plot_steps = len(FUNCTION_SPECS) * 2 + 1 # Cap plot steps to prevent infinite loop

        for _ in range(max_plot_steps):
            input_tensor = plot_residual.unsqueeze(0)
            output_logits = model(input_tensor)
            predicted_idx_plot = torch.argmax(output_logits, dim=1).item()

            if predicted_idx_plot == STOP_TOKEN_INDEX:
                break

            if predicted_idx_plot not in FIXED_FUNCTION_LIBRARY_INFO:
                break # Invalid prediction

            function_name = FIXED_FUNCTION_LIBRARY_INFO[predicted_idx_plot]
            func_spec = FUNCTION_SPECS[function_name]

            initial_guess = func_spec["initial_guess_strategy"]()
            bounds = func_spec["bounds"]

            target_signal_np = plot_residual.cpu().numpy()

            try:
                result = minimize(
                    objective_function,
                    initial_guess,
                    args=(target_signal_np, function_name),
                    method=OPTIMIZER_METHOD,
                    bounds=bounds,
                    options={'maxiter': MAX_OPTIMIZER_ITERATIONS, 'ftol': TOLERANCE}
                )
                optimized_params = result.x.tolist()
                optimized_component_signal_np = func_spec["func"](func_spec["params_gen_name"], optimized_params, SIGNAL_LENGTH)
                optimized_component_signal_torch = torch.tensor(optimized_component_signal_np, dtype=torch.float32).to(device)

                plot_residual -= optimized_component_signal_torch
                plot_reconstructed_signal += optimized_component_signal_torch
                plot_predicted_indices.append(predicted_idx_plot)
                plot_predicted_params.append(optimized_params)

            except Exception as e:
                # print(f"Plotting Optimizer failed for {function_name}: {e}. Skipping.")
                break # Stop processing this signal if optimizer fails

            if torch.norm(plot_residual) < 1e-3 and _ > 0: # Stop if residual is very small after extracting at least one component
                break

    print("\n***************")
    print("Graph Demonstration (Validation Example):")
    print(f"Actual Indices: {single_val_true_indices.cpu().numpy()}")
    print(f"Predicted Indices: {plot_predicted_indices}")
    print(f"Actual Params: {single_val_true_params}")
    print(f"Predicted Params: {plot_predicted_params}")
    print("***************\n")

    f = np.logspace(1, 6, SIGNAL_LENGTH)

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.semilogx(f, single_val_signal.cpu().numpy(), label="Original Signal (Dataset Example)", color='blue', alpha=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Original Signal from Validation Dataset")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    plt.subplot(4, 1, 2)
    plt.semilogx(f, plot_reconstructed_signal.cpu().numpy(), label="Reconstructed Signal (Classifier + Optimizer)", color='red', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Model's Reconstructed Signal (Sum of Optimized Components)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    plt.subplot(4, 1, 3)
    plt.semilogx(f, plot_residual.cpu().numpy(), label="Final Residual (Original - Reconstructed)", color='green')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Final Residual Signal")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    plt.subplot(4, 1, 4)
    if plot_predicted_indices:
        for i, idx in enumerate(plot_predicted_indices):
            # Ensure index is valid for plotting (might be invalid if optimizer failed)
            if idx in FIXED_FUNCTION_LIBRARY_INFO and i < len(plot_predicted_params):
                func_name = FIXED_FUNCTION_LIBRARY_INFO[idx]
                params = plot_predicted_params[i]
                if params is not None: # Only plot if params were successfully optimized
                    component_signal = FUNCTION_SPECS[func_name]["func"](FUNCTION_SPECS[func_name]["params_gen_name"], params, SIGNAL_LENGTH)
                    plt.semilogx(f, component_signal, label=f"Extracted {func_name} (Opt. Params)", alpha=0.7)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Individual Extracted Components (Optimized Parameters)")
        plt.legend()
        plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    else:
        plt.text(0.5, 0.5, "No components extracted by model.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate average losses
    avg_main_classification_loss = total_main_classification_loss / len(validation_dataloader)
    avg_reconstruction_loss = total_reconstruction_loss / len(validation_dataloader)
    
    return avg_main_classification_loss, avg_reconstruction_loss


# --- Section 7: Custom Collate Function for DataLoader ---

def custom_collate_fn(batch):
    """
    Collates a list of samples into a batch, handling variable-length sequences
    for indices and keeping parameters as a list of lists.
    """
    signals = [item[0] for item in batch]
    indices = [item[1] for item in batch]
    params = [item[2] for item in batch] # This will be a list of lists of parameters, or Nones

    # Stack signals into a single tensor
    signals_batch = torch.stack(signals)

    # Pad sequences of indices
    # Find the maximum sequence length in the batch
    max_len = max(len(seq) for seq in indices)
    # Pad all sequences to max_len with the STOP_TOKEN_INDEX
    padded_indices = []
    for seq in indices:
        padded_seq = torch.cat([seq, torch.full((max_len - len(seq),), STOP_TOKEN_INDEX, dtype=torch.long)])
        padded_indices.append(padded_seq)
    indices_batch = torch.stack(padded_indices)

    # Parameters can remain as a list of lists (Python objects).
    # DataLoader handles this well as long as it's not trying to convert it to a tensor.
    params_batch = params

    return signals_batch, indices_batch, params_batch


# --- Section 8: Main Execution Block ---

if __name__ == '__main__':
    # Declare global epoch variable here to avoid SyntaxError
    epoch = 0
    #global epoch 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Initialize global epoch for dataset generation logic
    epoch = 0 # Starts at 0, will be updated in the loop

    # 1. Create the full dataset (now generates variable parameters and progressive complexity)
    full_dataset = DecompositionDataset(num_samples=8000)

    # 2. Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. Create separate DataLoaders for training and validation, using the custom collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # 4. Model
    model = FunctionSelectorTransformer(
        signal_length=SIGNAL_LENGTH, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES,
        d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=N_LAYERS
    ).to(device)

    # Define both loss functions for training
    loss_fns = (nn.CrossEntropyLoss(reduction='none'), nn.MSELoss(reduction='sum'))

    # Define both weights for training
    weights = (STOP_LOSS_WEIGHT, RECONSTRUCTION_LOSS_WEIGHT)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5,) # Increased patience for stability

    # The `loss_fn` for evaluate_epoch's classification part
    loss_fn_eval = nn.CrossEntropyLoss()

    print("--- Starting Training ---")
    for current_epoch in range(EPOCHS):
        # Update the global epoch for the dataset's __getitem__
        epoch = current_epoch

        train_loss = train_epoch(model, train_dataloader, optimizer, loss_fns, device, weights)
        
        # In evaluate_epoch, the optimizer step is performed for reconstruction accuracy
        avg_val_classification_loss, avg_val_reconstruction_loss = evaluate_epoch(model, val_dataloader, loss_fn_eval, device)
        
        # Schedule based on reconstruction loss, as it reflects overall performance
        scheduler.step(avg_val_reconstruction_loss) 

        print(f"Epoch {current_epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Class Loss: {avg_val_classification_loss:.4f}, Val Recon Loss: {avg_val_reconstruction_loss:.4f}")

        # --- Save model checkpoint ---
        if (current_epoch + 1) >= SAVE_START_EPOCH and (current_epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            formatted_val_loss = f"{avg_val_reconstruction_loss:.4f}".replace('.', '_')
            model_filename = f"classifier_epoch{current_epoch+1}_val_recon_{formatted_val_loss}.pth"
            save_path = os.path.join(MODELS_DIR, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"Saved classifier model to {save_path}")
                
    print("--- Finished Training ---")