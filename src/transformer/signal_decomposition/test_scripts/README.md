# Filter Decomposition Test Scripts

This folder contains demonstration scripts for the filter response decomposition transformer model.

## Scripts Overview

### `test1.py` - Main Demonstration
The primary demonstration script that shows:
- **Top row**: Two input filter curves (lowpass and highpass) and their mixed signal
- **Bottom row**: Model predictions of the decomposed signals

**Usage:**
```bash
python test1.py
```

**Features:**
- Loads the trained model from `../output_models/`
- Generates sample lowpass (1kHz) and highpass (5kHz) filter responses
- Shows input signals and model predictions side-by-side
- Provides reconstruction quality analysis

### `demo_simple.py` - Clean Simple Demo
A streamlined version focused on the exact layout you requested:
- 2x3 grid layout
- Top row: input curves and mixed signal
- Bottom row: model predictions

### `interactive_demo.py` - Customizable Demo
Interactive script that allows you to:
- Test different cutoff frequencies
- Run multiple test cases
- Easily modify parameters in the script

## Quick Start

1. **Ensure model is trained**: Make sure `signal_decomposition_transformer.pth` exists in `../output_models/`

2. **Run the main demo**:
   ```bash
   cd test_scripts
   python test1.py
   ```

3. **For custom frequencies**, edit `interactive_demo.py`:
   ```python
   custom_lp = 800   # Change this value (Hz)
   custom_hp = 8000  # Change this value (Hz)
   ```

## What You'll See

The demonstration shows how the transformer neural network:

1. **Takes as input**: A mixed filter response (sum of lowpass + highpass)
2. **Produces as output**: Separate lowpass and highpass filter responses
3. **Validates quality**: By reconstructing the original mixed signal

### Example Filter Types

- **Lowpass**: `H(jω) = 1 / sqrt(1 + (ω/ωc)²)`
- **Highpass**: `H(jω) = (ω/ωc) / sqrt(1 + (ω/ωc)²)`
- **Mixed**: Simple addition of the two responses

### Frequency Range

- Input: 0.1 Hz to 1 MHz (logarithmic scale)
- Points: 128 input → 512 output (upsampled by model)

## Output

Each script generates:
- Interactive matplotlib plots
- Saved PNG files in the same directory
- Console analysis with reconstruction quality metrics

## Model Architecture

The transformer model used:
- Input: (batch, 128, 1) - mixed filter response
- Output: (batch, 2, 512) - [lowpass, highpass] responses
- Architecture: Transformer encoder with positional encoding and upsampling
