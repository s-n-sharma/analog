import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader

#todo: change loss function, it is ass rn, maybe transformers aren't the way??

# --- Section 1: Model Definition (Updated) ---

# PositionalEncoding class remains the same
class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class FunctionFinderTransformer(nn.Module):
    """
    The updated model that tokenizes the input signal into multiple patches.
    """
    def __init__(self, signal_length: int, patch_size: int, num_functions: int, max_params: int,
                 d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        assert signal_length % patch_size == 0, "Signal length must be divisible by patch size."
        
        num_patches = signal_length // patch_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 1. Input Embedding: Use a 1D convolution to create patches (tokens)
        # This is analogous to the 'conv' chunking strategy from the paper.
        self.patch_embedding = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 2. Positional Encoding for the sequence of patches
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=num_patches)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # 4. Multi-Task Head (unchanged)
        self.classification_head = nn.Linear(d_model, num_functions)
        self.regression_head = nn.Linear(d_model, max_params)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: The source signal tensor, shape [batch_size, signal_length]
        """
        # Add a channel dimension for the convolution: [batch_size, 1, signal_length]
        src = src.unsqueeze(1)
        
        # Create patches with the convolution: [batch_size, d_model, num_patches]
        src_patched = self.patch_embedding(src)
        
        # Permute for Transformer: [batch_size, num_patches, d_model]
        src_permuted = src_patched.permute(0, 2, 1)
        
        # Add positional encoding
        src_pos_encoded = self.pos_encoder(src_permuted)
        
        # Pass through the encoder
        encoded_patches = self.transformer_encoder(src_pos_encoded)
        
        # Aggregate the output tokens. Taking the mean is a common and effective strategy.
        # Shape: [batch_size, num_patches, d_model] -> [batch_size, d_model]
        aggregated_output = encoded_patches.mean(dim=1)
        
        # Pass the single aggregated vector to both heads
        classification_output = self.classification_head(aggregated_output)
        regression_output = self.regression_head(aggregated_output)
        
        return classification_output, regression_output

# --- Section 2: Helper Functions & Constants ---

SIGNAL_LENGTH = 400
MAX_PARAMS = 6 # Max params for any function

def generate_sine_signal(params, length):
    amp, freq, phase = params[0], params[1], params[2]
    x = np.linspace(0, 2 * np.pi, length)
    return amp * np.sin(freq * 5 * x + phase)

def generate_linear_signal(params, length):
    slope, intercept = params[0], params[1]
    return (slope * 0.1) * np.arange(length) + (intercept * 0.1)

#FUNCTION_MAP = {
#    0: {"name": "SINE", "generator": generate_sine_signal, "n_params": 3},
#    1: {"name": "LINEAR", "generator": generate_linear_signal, "n_params": 2},
#}

#from numpy import j

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
    r1, r2, c1, c2 = params[0], params[1], params[2], params[3]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
        num = 1
        denom = -1 * w * w * r1 * r2 * c1 * c2 + 1j * w * (r1 * c2 + r2*c2) + 1
        ret.append(np.abs(num/denom))
    return 20*np.log10(np.array(ret))

def sk_highpass(params, length):
    r1, r2, c1, c2 = params[0], params[1], params[2], params[3]
    x = np.logspace(1, 6, length)
    ret = []
    for w in x:
        num = -1 * w * w
        denom = -1 * w * w + 1j * w * (1/(r1 * c1) + 1/(r2 * c1)) + 1/(r1 * r2 * c1 * c2)
        ret.append(np.abs(num/denom))
    
    return 20*np.log10(np.array(ret))

def voltage_divider(params, length):
    r1, r2 = params[0], params[1]
    return 20*np.log10(r2/(r1 + r2) * np.ones(length))

FUNCTION_MAP = {
    0 : {"name" : "bandpass", "generator" : bandpass, "n_params" : 4},
    1 : {"name" : "inverting_amp", "generator" : inverting_amp, "n_params" : 2},
    #2 : {"name" : "non_inverting_amp", "generator" : non_inverting_amp, "n_params" : 2},
    3 : {"name" : "rc_highpass", "generator" : rc_highpass, "n_params" : 2},
    4 : {"name" : "rc_lowpass", "generator" : rc_lowpass, "n_params" : 2},
    5 : {"name" : "sallen_key_highpass", "generator" : sk_highpass, "n_params" : 4},
    6 : {"name" : "sallen_key_lowpass", "generator" : sk_lowpass, "n_params" : 4},
    7 : {"name" : "voltage_divider", "generator" : voltage_divider, "n_params" : 2},
}
# The STOP token is a special class the model learns to predict when decomposition is done
STOP_TOKEN_INDEX = len(FUNCTION_MAP)
NUM_FUNCTIONS = len(FUNCTION_MAP) + 1 # Add 1 for the STOP token


# --- Section 3: Dataset Simulation ---

class DummyDecompositionDataset(Dataset):
    """
    Simulates a dataset where each item is a signal and its ordered ground-truth decomposition.
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def err(self, a, b):
        return np.dot(a-b, a-b)

    def __getitem__(self, idx):
        # Create a random ground-truth decomposition
        # params1 = np.random.rand(3) * 2 - 1 # Sine params
        # params2 = np.random.rand(2) * 2 - 1 # Linear params
        
        # true_decomposition = [
        #     (0, torch.tensor(params1, dtype=torch.float32)), # (class_idx, params)
        #     (1, torch.tensor(params2, dtype=torch.float32))
        # ]
        
        # # Generate the signal from the decomposition
        # signal1 = generate_sine_signal(params1, SIGNAL_LENGTH)
        # signal2 = generate_linear_signal(params2, SIGNAL_LENGTH)
        # noise = np.random.randn(SIGNAL_LENGTH) * 0.1
        # final_signal = torch.tensor(signal1 + signal2 + noise, dtype=torch.float32)
        
        # return final_signal, true_decomposition

        import random 


        function_number = random.randint(1, 1)
        functions = random.choices(list(range(len(FUNCTION_MAP))), k=function_number)

        common_resistor_values_ohms = [10, 22, 33, 47, 68, 100, 220, 330, 470, 680, 1000, 2200, 3300, 4700, 6800, 10000, 22000, 33000, 47000, 68000, 100000, 220000, 330000, 470000, 680000, 1000000]
        common_capacitor_values_farads = [10e-12, 22e-12, 33e-12, 47e-12, 68e-12, 100e-12, 220e-12, 330e-12, 470e-12, 680e-12, 1e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 10e-9, 22e-9, 33e-9, 47e-9, 68e-9, 100e-9, 220e-9, 330e-9, 470e-9, 680e-9, 1e-6, 2.2e-6, 3.3e-6, 4.7e-6, 10e-6, 22e-6, 33e-6, 47e-6, 100e-6, 220e-6, 330e-6, 470e-6, 1000e-6]

        sig = np.zeros(400, dtype=np.float32)
        f = np.logspace(1, 6, 400, dtype=np.float32)
        func_list = []
        # fucked = False
        for func in functions:
            if func == 0: #bandpass
                r1, r2, c1, c2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads), random.choice(common_capacitor_values_farads)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, c1, r2, c2], 400)
                func_list.append((0, torch.tensor([r1, c1, r2, c2], dtype=torch.float32)))
            if func == 1: #inverting_amp
                r1, r2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, r2], 400)
                func_list.append((1, torch.tensor([r1, r2], dtype=torch.float32)))
            if func == 2: #non_inverting_amp
                continue
                r1, r2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, r2], 400)
                func_list.append((2, torch.tensor([r1, r2], dtype=torch.float32)))
            if func == 3: #rc_highpass
                r, c = random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads)
                sig = sig + FUNCTION_MAP[func]["generator"]([r, c], 400)
                func_list.append((3, torch.tensor([r, c], dtype=torch.float32)))
            if func == 4: #rc_lowpass
                r, c = random.choice(common_resistor_values_ohms), random.choice(common_capacitor_values_farads)
                sig = sig + FUNCTION_MAP[func]["generator"]([r, c], 400)
                func_list.append((4, torch.tensor([r, c], dtype=torch.float32)))
            if func == 5: #sallen_key_highpass
                r1, r2, c1, c2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, r2, c1, c2], 400)
                func_list.append((5, torch.tensor([r1, r2, c1, c2], dtype=torch.float32)))
            if func == 6: #sallen_key_lowpass
                r1, r2, c1, c2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, r2, c1, c2], 400)
                func_list.append((6, torch.tensor([r1, r2, c1, c2], dtype=torch.float32)))
            if func == 7: #voltage_divider
                r1, r2 = random.choice(common_resistor_values_ohms), random.choice(common_resistor_values_ohms)
                sig = sig + FUNCTION_MAP[func]["generator"]([r1, r2], 400)
                func_list.append((7, torch.tensor([r1, r2], dtype=torch.float32)))
            # try:
            #     if not fucked:
            #         print(len(sig[0]))
            #         print("sig is not 2d", func)
            #         fucked = not fucked
            # except:
            #     print("failed")
        
        # ret = torch.from_numpy(sig)
        # print(ret)      
        temp = sorted(func_list, key=lambda x: self.err(sig, FUNCTION_MAP[x[0]]["generator"](x[1].numpy(), 400)))
        temp.append((7, torch.tensor([], dtype=torch.float32)))
        return (torch.tensor(sig, dtype=torch.float32), temp)









def collate_fn(batch):
    """Custom collate function to handle lists of varying lengths."""
    signals, decompositions = zip(*batch)
    return torch.stack(signals), list(decompositions)

# --- Section 4: The Training Function ---

def train_epoch(model, dataloader, optimizer, class_loss_fn, reg_loss_fn, device,
                regression_loss_weight, stop_loss_weight):
    """
    The training function, updated with explicit loss weights to control model incentives.
    """
    model.train()
    total_epoch_loss = 0

    for signals, true_decompositions in dataloader:
        signals = signals.to(device)
        optimizer.zero_grad()

        batch_loss = torch.tensor(0.0, device=device)

        for i in range(len(signals)):
            signal_item = signals[i]
            true_decomp_item = true_decompositions[i]

            current_residual = signal_item.clone()

            for true_class_idx, true_params in true_decomp_item:
                pred_logits, pred_params = model(current_residual.unsqueeze(0))
                true_class_tensor = torch.tensor([true_class_idx], device=device)

                # Check if the current step is the STOP token
                if true_class_idx == STOP_TOKEN_INDEX:
                    # --- FIX IS HERE: Apply the STOP_LOSS_WEIGHT ---
                    # Calculate the stop loss and scale it by its specific weight.
                    loss_stop = class_loss_fn(pred_logits, true_class_tensor)
                    batch_loss = batch_loss + (stop_loss_weight * loss_stop)
                    break # Exit the loop for this item

                # --- This code now only runs for NON-STOP tokens ---
                # --- FIX IS HERE: Apply the REGRESSION_LOSS_WEIGHT ---
                loss_c = class_loss_fn(pred_logits, true_class_tensor)

                n_params = FUNCTION_MAP[true_class_idx]["n_params"]
                padded_true_params = torch.zeros(MAX_PARAMS, device=device)
                num_padding_zeros = MAX_PARAMS - n_params
                zeros_to_pad = torch.zeros(num_padding_zeros, device=device)
                padded_true_params = torch.cat([true_params.to(device), zeros_to_pad])
                
                mask = torch.zeros(MAX_PARAMS, device=device)
                mask[:n_params] = 1.0
                loss_r = (reg_loss_fn(pred_params.squeeze(0), padded_true_params) * mask).sum() / mask.sum()

                # Combine the step loss using the regression weight
                step_loss = loss_c + (regression_loss_weight * loss_r)
                batch_loss = batch_loss + step_loss

                # Update residual for the next step
                true_signal_component = FUNCTION_MAP[true_class_idx]["generator"](true_params.numpy(), SIGNAL_LENGTH)
                current_residual = current_residual - torch.tensor(true_signal_component, dtype=torch.float32, device=device)


        normalized_batch_loss = batch_loss / len(signals)
        normalized_batch_loss.backward()
        optimizer.step()

        total_epoch_loss += normalized_batch_loss.item()

    return total_epoch_loss / len(dataloader)

# --- Section 5: The Inference Class ---

class GenerativeDecomposer:
    """Uses the trained model to decompose a signal autoregressively."""
    def __init__(self, model, function_map, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.function_map = function_map
        self.device = device

    @torch.no_grad()
    def decompose(self, signal, max_steps=5, stop_threshold=0.05):
        residual_signal = signal.clone().to(self.device)
        decomposition = []
        
        for _ in range(max_steps):
            if torch.mean(residual_signal**2).item() < stop_threshold:
                print("Stopping: Residual energy below threshold.")
                break
            
            pred_logits, pred_params = self.model(residual_signal.unsqueeze(0))
            pred_class_idx = torch.argmax(pred_logits, dim=1).item()
            
            if pred_class_idx == STOP_TOKEN_INDEX:
                print("Stopping: Model predicted STOP token.")
                break
            
            func_info = self.function_map.get(pred_class_idx)
            if not func_info:
                print(f"Stopping: Predicted unknown class {pred_class_idx}.")
                break
                
            n_params = func_info["n_params"]
            relevant_params = pred_params[0, :n_params].cpu().numpy()
            decomposition.append((func_info["name"], relevant_params))
            
            signal_component = func_info["generator"](relevant_params, SIGNAL_LENGTH)
            residual_signal -= torch.tensor(signal_component, dtype=torch.float32, device=self.device)
            
        return decomposition


# --- Section 6: Main Execution Block ---

if __name__ == '__main__':
    # Hyperparameters
    # In your main execution block (if __name__ == "__main__":)

    # --- Hyperparameters ---
    PATCH_SIZE = 20
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 3
    EPOCHS = 5 # Increased epochs might be needed
    LR = 0.00001 # A slightly lower learning rate is often more stable
    BATCH_SIZE = 8

    # --- NEW LOSS WEIGHT HYPERPARAMETERS ---
    # Weight for the parameter regression loss.
    # A value less than 1.0 treats it as an auxiliary task to the main classification.
    REGRESSION_LOSS_WEIGHT = 0.5

    # Weight for the final STOP token's classification loss.
    # A small value tells the model that getting the main decomposition steps
    # right is much more important than the final stop action.
    STOP_LOSS_WEIGHT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    train_dataset = DummyDecompositionDataset(num_samples=4000)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 2. Model
    model = FunctionFinderTransformer(
        signal_length=SIGNAL_LENGTH,
        patch_size=PATCH_SIZE,  # New argument
        num_functions=NUM_FUNCTIONS,
        max_params=MAX_PARAMS,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=N_LAYERS
    ).to(device)

    # 3. Loss and Optimizer
    class_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss(reduction='none') # Use 'none' for masking
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    # In your main execution block's training loop
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
    # --- FIX IS HERE: Pass the new weights to the function ---
        avg_loss = train_epoch(
        model,
        train_dataloader,
        optimizer,
        class_loss_fn,
        reg_loss_fn,
        device, 
        regression_loss_weight=REGRESSION_LOSS_WEIGHT,
        stop_loss_weight=STOP_LOSS_WEIGHT
        )
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    print("--- Finished Training ---")

    # 5. Inference Example
    print("\n--- Running Inference Example ---")
    # Create a new test signal
    # test_params1 = np.array([0.8, 1.5, 1.0]) # Sine
    # test_params2 = np.array([-1.2, 0.5])   # Linear
    # test_signal_numpy = generate_sine_signal(test_params1, SIGNAL_LENGTH) + \
    #                     generate_linear_signal(test_params2, SIGNAL_LENGTH)
    # test_signal_tensor = torch.tensor(test_signal_numpy, dtype=torch.float32)

    test1 = train_dataset.__getitem__(10)
    test1_signal = test1[0]

    # Use the trained model to decompose it
    decomposer = GenerativeDecomposer(model, FUNCTION_MAP, device)
    predicted_decomposition = decomposer.decompose(test1_signal)
    
    print("\n--- Final Decomposition ---")
    #print(f"Ground Truth: SINE({test_params1}), LINEAR({test_params2})")
    print("Actual Circuit")
    for thing in test1[1]:
        try:
            print(FUNCTION_MAP[thing[0]]["name"], thing[1])
        except:
            print("stop token")
    if predicted_decomposition:
        for name, params in predicted_decomposition:
            print(f"Predicted: {name}({np.round(params, 2)})")