import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize # <--- NEW IMPORT

# --- Section 1: Constants and Configuration (MUST MATCH TRAINING CONFIG) ---

# Signal and Model Hyperparameters
SIGNAL_LENGTH = 400
PATCH_SIZE = 20
D_MODEL = 256
N_HEAD = 16
N_LAYERS = 8

# Loss Weight for the STOP token (not used in inference, but part of config)
STOP_LOSS_WEIGHT = 1.2

# Model Loading Configuration
MODELS_DIR = "saved_models"
MODEL_FILENAME = "model_epoch20_val_loss_1_0587.pth" # <--- REPLACE WITH YOUR ACTUAL MODEL FILENAME
                                                   # Make sure this is a model trained with the *variable parameter* dataset logic if you want to test that.

# --- NEW: Optimizer Configuration ---
OPTIMIZER_METHOD = 'L-BFGS-B' # A good general-purpose method for bounded, unconstrained problems
MAX_OPTIMIZER_ITERATIONS = 50 # Limit iterations for speed
TOLERANCE = 1e-6 # Tolerance for optimization convergence
# --------------------------------------------------------------------------


# --- Section 2: Function Definitions and Parameter Ranges ---
# All your signal generation functions: bandpass, inverting_amp, etc.
# These must be identical to your training script.

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


# --- Parameter ranges for each function type ---
# These are crucial for the optimizer
common_resistor_values_ohms = [10, 22, 33, 47, 68, 100, 220, 330, 470, 680, 1000, 2200, 3300, 4700, 6800, 10000, 22000, 33000, 47000, 68000, 100000, 220000, 330000, 470000, 680000, 1000000]
common_capacitor_values_farads = [10e-12, 22e-12, 33e-12, 47e-12, 68e-12, 100e-12, 220e-12, 330e-12, 470e-12, 680e-12, 1e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9, 220e-9, 330e-9, 470e-9, 680e-9, 1e-6, 2.2e-6, 3.3e-6, 4.7e-6, 10e-6, 22e-6, 33e-6, 47e-6, 100e-6, 220e-6, 330e-6, 470e-6, 1000e-6]
common_cutoff_values = np.logspace(1, 6, 100) # More values for optimization, from 10Hz to 1MHz
common_q_values = [0.5, 0.707, 1.0, 1.414, 2.0, 3.0, 5.0, 10.0] # More Q values

# Map function names to their actual generation functions and parameter bounds
FUNCTION_SPECS = {
    "inverting_amp": {
        "func": inverting_amp,
        "num_params": 2,
        "bounds": [(10, 1e6), (10, 1e6)], # R1, R2 ranges
        "initial_guess_strategy": lambda: [random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)]
    },
    "sk_highpass": {
        "func": sk_highpass,
        "num_params": 2,
        "bounds": [(10, 1e6), (0.1, 10.0)], # w0 (cutoff freq), Q (damping)
        "initial_guess_strategy": lambda: [random.choice(common_cutoff_values), random.choice(common_q_values)]
    },
    "sk_lowpass": {
        "func": sk_lowpass,
        "num_params": 2,
        "bounds": [(10, 1e6), (0.1, 10.0)], # w0 (cutoff freq), Q (damping)
        "initial_guess_strategy": lambda: [random.choice(common_cutoff_values), random.choice(common_q_values)]
    },
    # Add other functions here as needed
    # "bandpass": { ... }
}

# Define FIXED_FUNCTION_LIBRARY as a lookup for names, not pre-computed signals
# This is what the classifier predicts
FIXED_FUNCTION_LIBRARY_INFO = {
    0: "inverting_amp",
    1: "sk_highpass",
    2: "sk_lowpass",
}
STOP_TOKEN_INDEX = len(FIXED_FUNCTION_LIBRARY_INFO)
NUM_CLASSES = len(FIXED_FUNCTION_LIBRARY_INFO) + 1 # Add 1 for the STOP token


# --- Section 3: Model Architecture (MUST MATCH TRAINING CONFIG) ---

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


# --- Section 4: Dataset (Adapted for variable parameters) ---
# Global epoch variable for dataset generation logic
global epoch
epoch = 200 # Set a high epoch to trigger multi-component generation

curr_epoch = 5 # Used internally by the dataset __getitem__
idx_count = 3 # Used internally by the dataset __getitem__

class DecompositionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx, check=False):
        global epoch # Accessing global epoch here

        num_function_types = len(FIXED_FUNCTION_LIBRARY_INFO)

        # Determine number of components based on epoch
        if epoch < 2:
            num_components_to_sum = 1
        elif epoch < 14:
            num_components_to_sum = 1 # Can add a distractor in training, but for clean test, focus on main components
        else: # epoch >= 14, general multi-component logic
            max_allowed_components = min(3, num_function_types) # e.g., max 3 unique functions
            num_components_to_sum = np.random.randint(1, max_allowed_components + 1)
            # Use random.choices if you want to allow repeated function *types*
            # For unique types, use random.sample
            # For simplicity, let's allow repeated types for now.
        
        # Randomly choose function types (allowing repetition)
        chosen_base_indices = random.choices(list(FIXED_FUNCTION_LIBRARY_INFO.keys()), k=num_components_to_sum)
        
        signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32)
        true_decomposition_indices = []
        true_decomposition_params = [] # To store the actual parameters used for generation

        for func_idx in chosen_base_indices:
            func_name = FIXED_FUNCTION_LIBRARY_INFO[func_idx]
            func_spec = FUNCTION_SPECS[func_name]
            
            # Generate random parameters within the defined bounds for this function type
            # For simplicity, let's just pick a random value within the range for each param
            params = []
            for lower_bound, upper_bound in func_spec["bounds"]:
                # Adjust for log-scale parameters like w0 (cutoff frequency)
                if func_name in ["sk_highpass", "sk_lowpass"] and len(params) == 0: # w0
                     # Sample from log-uniform distribution for frequency
                    param_val = np.exp(np.random.uniform(np.log(lower_bound), np.log(upper_bound)))
                else:
                    param_val = np.random.uniform(lower_bound, upper_bound)
                params.append(param_val)

            # Generate the signal component using the *randomly chosen parameters*
            component_signal = torch.tensor(func_spec["func"](params, SIGNAL_LENGTH), dtype=torch.float32)
            signal += component_signal

            true_decomposition_indices.append(func_idx)
            true_decomposition_params.append(params) # Store the actual parameters

        # Sort the ground truth indices and their corresponding parameters for consistency
        # This requires a custom sort that keeps params paired with indices
        combined_truth = sorted(zip(true_decomposition_indices, true_decomposition_params), key=lambda x: x[0])
        true_decomposition_indices = [item[0] for item in combined_truth]
        true_decomposition_params = [item[1] for item in combined_truth]
        
        true_decomposition_indices.append(STOP_TOKEN_INDEX)
        true_decomposition_params.append(None) # No parameters for STOP token

        if check:
            return signal, torch.tensor(true_decomposition_indices, dtype=torch.long), true_decomposition_params
        return signal, torch.tensor(true_decomposition_indices, dtype=torch.long)


# --- Section 5: Optimization Objective Function ---

def objective_function(params, target_signal_np, function_name):
    """
    Objective function for scipy.optimize.minimize.
    Calculates MSE between the target_signal and a signal generated with given parameters.

    Args:
        params (np.ndarray): Array of parameters for the function.
        target_signal_np (np.ndarray): The NumPy array of the residual signal.
        function_name (str): The name of the function to generate (e.g., "sk_highpass").

    Returns:
        float: Mean Squared Error.
    """
    # Ensure params are within float32 range if you eventually pass them to torch
    # For scipy, float64 is default, which is fine here.
    generated_signal = FUNCTION_SPECS[function_name]["func"](params.tolist(), SIGNAL_LENGTH)
    mse = np.mean((generated_signal - target_signal_np)**2)
    return mse

# --- Main Loading and Inference Logic ---

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Construct the full path to the saved model
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please check your MODELS_DIR and MODEL_FILENAME constants.")
        if os.path.exists(MODELS_DIR):
            print(f"Files found in '{MODELS_DIR}': {os.listdir(MODELS_DIR)}")
        exit()

    # Instantiate the model
    model = FunctionSelectorTransformer(
        signal_length=SIGNAL_LENGTH, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES,
        d_model=D_MODEL, nhead=N_HEAD, num_encoder_layers=N_LAYERS
    ).to(device)

    # Load the saved state_dict
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode (important for dropout, batchnorm etc.)
    print("Model loaded successfully.")

    # --- Load an example from the dataset with variable parameters ---
    print("\n--- Loading example from Dataset with variable parameters ---")
    # We use 'epoch = 100' globally to ensure multi-component signals are generated
    test_dataset = DecompositionDataset(num_samples=1)
    original_input_signal, true_decomposition_indices_tensor, true_decomposition_params_list = test_dataset.__getitem__(0, check=True)
    original_input_signal = original_input_signal.to(device) # Move to device
    true_decomposition_indices_np = true_decomposition_indices_tensor.cpu().numpy()


    print(f"Original Signal (from dataset) Min: {original_input_signal.min():.2f}, Max: {original_input_signal.max():.2f}")
    print(f"Ground Truth Decomposition Indices: {true_decomposition_indices_np}")
    print(f"Ground Truth Decomposition Parameters: {true_decomposition_params_list}")

    current_residual = original_input_signal.clone()
    predicted_component_info = [] # Stores (index, optimized_params)
    reconstructed_signal = torch.zeros(SIGNAL_LENGTH, dtype=torch.float32, device=device)

    print("\n--- Running Inference with Classifier + Optimizer ---")

    MAX_EXTRACTION_STEPS = len(FUNCTION_SPECS) * 2 + 1 # Allow more steps if components might be repeated

    with torch.no_grad(): # Classifier part is no_grad
        for step in range(MAX_EXTRACTION_STEPS):
            input_tensor = current_residual.unsqueeze(0) # Add batch dimension
            output_logits = model(input_tensor)
            predicted_index_classifier = torch.argmax(output_logits, dim=1).item()

            if predicted_index_classifier == STOP_TOKEN_INDEX:
                print(f"Step {step+1}: Classifier predicted STOP token. Ending decomposition.")
                break

            if predicted_index_classifier not in FIXED_FUNCTION_LIBRARY_INFO:
                print(f"Step {step+1}: Classifier predicted invalid index {predicted_index_classifier}. Stopping.")
                break

            function_name = FIXED_FUNCTION_LIBRARY_INFO[predicted_index_classifier]
            func_spec = FUNCTION_SPECS[function_name]

            print(f"Step {step+1}: Classifier predicted component type: {function_name} (Index: {predicted_index_classifier})")

            # --- Optimization Part ---
            initial_guess = func_spec["initial_guess_strategy"]()
            bounds = func_spec["bounds"]

            # Convert current_residual to NumPy for SciPy's optimizer
            target_signal_np = current_residual.cpu().numpy()

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
                optimized_loss = result.fun
                print(f"   Optimizer found params: {np.array(optimized_params).round(2).tolist()} with MSE: {optimized_loss:.4f}")

                # Generate the component signal with the OPTIMIZED parameters
                optimized_component_signal_np = func_spec["func"](optimized_params, SIGNAL_LENGTH)
                optimized_component_signal_torch = torch.tensor(optimized_component_signal_np, dtype=torch.float32).to(device)

                # Update the residual using the optimized component
                current_residual = current_residual - optimized_component_signal_torch
                reconstructed_signal = reconstructed_signal + optimized_component_signal_torch
                predicted_component_info.append({"index": predicted_index_classifier, "params": optimized_params})

            except Exception as e:
                print(f"   Optimizer failed for {function_name}: {e}. Skipping this component.")
                # If optimizer fails, we might still want to subtract the component based on default params
                # or just break, depending on desired robustness. For now, we continue.
                break # Break if optimization fails to prevent issues


            # Optional: Check if residual is very small, might indicate termination
            if torch.norm(current_residual) < 1e-3 and step > 0: # Small norm, and has extracted at least one component
                print(f"Residual became very small ({torch.norm(current_residual):.4e}). Ending decomposition.")
                break

    print(f"\n--- Decomposition Summary ---")
    print(f"Original Signal (from dataset) Min: {original_input_signal.min():.2f}, Max: {original_input_signal.max():.2f}")
    print(f"Ground Truth Decomposition Indices: {true_decomposition_indices_np}")
    print(f"Ground Truth Decomposition Parameters: {true_decomposition_params_list}")
    
    predicted_indices_summary = [info['index'] for info in predicted_component_info]
    predicted_params_summary = [info['params'] for info in predicted_component_info]
    print(f"Final Predicted Component Sequence (Indices): {predicted_indices_summary}")
    print(f"Final Predicted Component Parameters: {predicted_params_summary}")

    print(f"Final residual norm: {torch.norm(current_residual):.4e}")
    # The reconstruction error should now be much lower IF the model classifies correctly AND the optimizer works well.
    final_reconstruction_error = nn.functional.mse_loss(reconstructed_signal, original_input_signal)
    print(f"Final Reconstruction Error (MSE): {final_reconstruction_error:.4f}")

    # --- Visualization ---
    f = np.logspace(1, 6, SIGNAL_LENGTH)

    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.semilogx(f, original_input_signal.cpu().numpy(), label="Original Dataset Signal (dB)", color='blue', alpha=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Original Signal from Dataset (Variable Parameters)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    plt.subplot(4, 1, 2)
    plt.semilogx(f, reconstructed_signal.cpu().numpy(), label="Reconstructed Signal (Classifier + Optimizer)", color='red', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Model's Reconstructed Signal (Sum of Optimized Components)")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    plt.subplot(4, 1, 3)
    plt.semilogx(f, current_residual.cpu().numpy(), label="Final Residual (Original - Reconstructed)", color='green')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Final Residual Signal")
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)

    # Plot individual optimized components
    plt.subplot(4, 1, 4)
    if predicted_component_info:
        for i, comp_info in enumerate(predicted_component_info):
            func_name = FIXED_FUNCTION_LIBRARY_INFO[comp_info['index']]
            optimized_comp_signal = FUNCTION_SPECS[func_name]["func"](comp_info['params'], SIGNAL_LENGTH)
            plt.semilogx(f, optimized_comp_signal, label=f"Extracted {func_name} ({i+1})", alpha=0.7)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Individual Extracted Components (Optimized Parameters)")
        plt.legend()
        plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    else:
        plt.text(0.5, 0.5, "No components extracted.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.axis('off')

    plt.tight_layout()
    plt.show()