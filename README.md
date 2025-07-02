# Analog AI

A comprehensive toolkit for analog circuit simulation, generation, and AI-powered analysis. This project combines traditional circuit analysis with modern machine learning approaches for signal processing and circuit synthesis.

## Architecture Overview

The codebase consists of three main components that work together to provide end-to-end circuit analysis capabilities:

### 1. Circuit Simulation Engine (`src/circuit/`)
- **Circuit.py**: Core circuit simulator using Modified Nodal Analysis (MNA)
- **translator.py**: SPICE-like netlist parser with subcircuit and function support
- **plotter.py**: Frequency response visualization (Bode plots)
- **optimization.py**: Interactive circuit parameter optimization
- **circuitgen.py**: Random circuit topology generation

### 2. Signal Decomposition Transformer (`src/transformer/signal_decomposition/`)
- **Transformer model**: Decomposes mixed filter responses into constituent components
- **Filter generation**: Creates realistic lowpass/highpass filter responses in dB
- **Training pipeline**: End-to-end training with normalization and caching
- **Test scripts**: Comprehensive evaluation and demo tools

### 3. Neural Circuit Synthesis (`src/transformer/neural_network/`)
- **Signal-to-circuit mapping**: Converts frequency responses to circuit parameters
- **Functional circuit generation**: Learns to synthesize parametric circuits
- **Data generation**: Creates training datasets for various circuit types

## Circuit Blocks
### How to write circuits

1. **Declare voltage sources** (any line that starts with `V`)  
   ```spice
   Vin <n_node_no.> <p_node_no.> <value>
   ```

2. **Declare components** (`R` = resistor, `C` = capacitor, `IOP` = Ideal op amp, `X` = name of sub circuit)  
   ```spice
   R1 <n_node_no.> <p_node_no.> <value>
   C1 <n_node_no.> <p_node_no.> <value>
   IOP <V+_no.> <V-_no.> <Vout_no.>
   X <start_node> <end_node> name
   ```

3. **Specify the output node**  
   ```spice
   Vout <n_node_no.> 
   ```

4. **End the netlist**  
   ```spice
   .end
   ```
### Subcircuits
Fixed-component circuit blocks defined in `.subckts`:
```spice
* In .subckts/example.sp:
.declare_subckt in out
R1 in out 1k
R2 out 0 2k

* In main circuit:
X1 1 2 example
```

### Functions
Parameterized circuit blocks defined in `.functions`:
```spice
* In .functions/example.sp:
* Parameters: R1, R2
R1 in out R1
R2 out 0 R2

* In main circuit:
.fn 1 2 example R1=1k R2=2k
```

## Usage

1. **Subcircuits**: Use `X` prefix for fixed blocks
2. **Functions**: Use `.fn` for parameterized blocks
3. **Interface**: 
   - Subcircuits: `.declare_subckt in out`
   - Functions: Use `in` and `out` nodes

## Examples

### Cascaded Filters
```spice
* Input source
V1 1 0 1

* Low-pass filter
.fn 1 2 rc_lowpass R=1k C=1u

* Band-pass filter
.fn 2 3 bandpass R1=1k C1=1u R2=10k C2=0.1u

* Output amplifier
.fn 3 4 non_inverting_amp R1=1k R2=10k
```

### Mixed Usage
```spice
* Function with parameters
Vin 0 1 5

.fn 1 2 voltage_divider R1=10k R2=20k

* Fixed subcircuit
X1 2 3 fixed_filter

Vout 3
.end
```

## Best Practices

1. Document parameters and circuit behavior
2. Use standard units (k, u, n, p)
3. Test with various parameter values
4. Verify node connections

Available functions: `rc_lowpass`, `rc_highpass`, `bandpass`, `voltage_divider`, `inverting_amp`, `non_inverting_amp`, `sallen_key_lowpass`, `sallen_key_highpass`

## 📁 Project Structure

```
analog/
├── src/
│   ├── circuit/                    # Circuit simulation engine
│   │   ├── .functions/            # Parameterized circuit blocks  
│   │   ├── .subckts/              # Fixed subcircuits
│   │   ├── Circuit.py             # MNA solver
│   │   ├── translator.py          # Netlist parser
│   │   ├── plotter.py             # Bode plot generation
│   │   ├── optimization.py        # Parameter optimization
│   │   └── circuitgen.py          # Random circuit generation
│   └── transformer/               # AI components
│       ├── signal_decomposition/  # Filter response decomposition
│       │   ├── data/              # Filter response generation
│       │   ├── model/             # Transformer architecture
│       │   ├── train.py           # Training pipeline
│       │   └── test_scripts/      # Evaluation and demos
│       └── neural_network/        # Circuit synthesis
│           ├── model.py           # Neural architecture
│           └── functional_*.py    # Circuit parameter generation
└── README.md
```

## Current Status & TODO

### Completed
- **Circuit Simulation**: Complete MNA solver with SPICE-like netlist parsing
- **Subcircuits & Functions**: Modular circuit blocks (8 function types available)
- **Signal Decomposition**: Working transformer model with <1.1 dB reconstruction error
- **Filter Response Generation**: Realistic lowpass/highpass filters in dB domain
- **Interactive Optimization**: Draw target response and optimize circuit parameters

### In Progress  
- **Neural Circuit Synthesis**: Basic framework exists for mapping responses to parameters
- **Advanced Optimization**: Gradient-based parameter tuning for circuit components

### TODO: System Integration
The ultimate goal is to link all three components into a unified circuit design pipeline:

1. **User Input** → Target frequency response (drawn/specified)
2. **Signal Decomposition** → Break complex response into basic filter components  
3. **Circuit Synthesis** → Map decomposed components to actual circuit topologies
4. **Parameter Optimization** → Fine-tune component values using gradient descent
5. **Validation** → Simulate final circuit and verify performance matches target

**Integration Points:**
- Connect signal decomposition output to circuit synthesis input
- Link circuit synthesis to optimization for parameter refinement
- Create unified API for end-to-end design flow
- Add feedback loop for iterative improvement

This would enable **automatic circuit design from high-level specifications** - a user could simply draw a desired frequency response and get a complete, optimized circuit implementation.

## Performance Metrics

**Signal Decomposition Transformer:**
- Model: 768-dim, 12-head, 4-layer transformer
- Training data: 10k samples of mixed filter responses  
- Performance: <1.1 dB average reconstruction error
- Input: 128-point frequency response → Output: 512-point decomposed signals

**Circuit Simulation:**
- Supports complex impedance analysis across frequency ranges
- Handles up to 10+ node circuits with multiple components
- Compatible with standard SPICE component models

## AI Components

### Signal Decomposition Transformer
Decomposes mixed frequency responses into constituent filter components:

```python
# Generate filter responses (in dB)
from transformer.signal_decomposition.data.generation import generate_signal1, generate_signal2

lowpass = generate_signal1(cutoff_freq=1000, length=128)    # 1kHz lowpass
highpass = generate_signal2(cutoff_freq=5000, length=128)   # 5kHz highpass
mixed = lowpass + highpass  # Combined response in dB domain

# Model predicts original components from mixed signal
model = SignalDecompositionTransformer(seq_len=128, d_model=768, ...)
predicted_lowpass, predicted_highpass = model(mixed)
```

**Key Features:**
- Works in log-magnitude (dB) domain for realistic filter responses
- Advanced transformer architecture with positional encoding
- Upsampling to higher resolution outputs (128→512 points)
- Normalization for training stability

### Neural Circuit Synthesis
Maps frequency responses to circuit parameters:

```python
# Generate RC lowpass data: frequency response → R,C values
from transformer.neural_network.model import FunctionalDataGeneration

data_gen = FunctionalDataGeneration("RC_lowpass", num_samples=50000, ...)
X_responses, y_parameters = data_gen.generate_data()

# Train neural network to predict circuit parameters
model = CircuitSynthesisNet(...)
predicted_params = model(frequency_response)
```

## 🚀 Usage Examples

### 1. Basic Circuit Simulation
```python
from circuit.translator import Translator
from circuit.plotter import Plotter

# Parse and simulate circuit
translator = Translator("my_circuit.sp")
circuit = translator.circuit

# Generate frequency response
plotter = Plotter(circuit, output_node=2, freq_range=(1, 1e5))
plotter.plot_magnitude()
```

### 2. Interactive Circuit Optimization
```python
from circuit.optimization import InteractiveLogDraw

# Draw target response, optimize circuit to match
optimizer = InteractiveLogDraw(x_sample_points=np.logspace(1, 6, 100))
# Draw on plot, press 's' to sample, then close to start optimization
```

### 3. AI-Powered Signal Analysis
```bash
# Train signal decomposition model
cd src/transformer/signal_decomposition
python train.py

# Test decomposition on new signals  
python test_scripts/test1.py
python test_scripts/interactive_demo.py
```

