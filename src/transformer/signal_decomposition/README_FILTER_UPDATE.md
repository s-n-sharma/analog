# Filter-Based Signal Decomposition

This module has been updated to work with electronic filter responses instead of trigonometric functions. The transformer now learns to decompose mixed frequency responses into their constituent lowpass and highpass filter components.

## Key Changes

### Signal Generation (`data/generation.py`)

1. **`generate_signal1(cutoff_freq, length)`**: Now generates lowpass filter magnitude response
   - Formula: `|H(jω)| = 1 / sqrt(1 + (ω/ωc)²)`
   - Input: cutoff frequency in Hz
   - Output: magnitude response over frequency range 0.1 Hz to 1 MHz

2. **`generate_signal2(cutoff_freq, length)`**: Now generates highpass filter magnitude response
   - Formula: `|H(jω)| = (ω/ωc) / sqrt(1 + (ω/ωc)²)`
   - Input: cutoff frequency in Hz
   - Output: magnitude response over frequency range 0.1 Hz to 1 MHz

3. **Updated parameter names**: 
   - `a`, `b` → `fc1`, `fc2` (cutoff frequencies)
   - `a_range`, `b_range` → `fc1_range`, `fc2_range`
   - Default ranges changed to realistic frequency values (10 Hz - 100 kHz)

### Additional Filter Types

- **`generate_bandpass_filter()`**: Creates bandpass response between two cutoff frequencies
- **`generate_notch_filter()`**: Creates notch (band-stop) filter with center frequency and Q factor
- **`generate_filter_with_phase()`**: Returns both magnitude and phase responses
- **`add_noise_to_signal()`**: Adds realistic noise to filter responses

### Training Script Updates (`train.py`)

- Updated comments to reflect filter response decomposition
- Variable names changed from `s1`, `s2` to `lowpass`, `highpass` for clarity
- Cache file still maintains backward compatibility

## Usage Examples

### Basic Filter Generation
```python
from data.generation import generate_signal1, generate_signal2, generate_sample

# Generate individual filter responses
lowpass = generate_signal1(cutoff_freq=1000, length=1000)  # 1 kHz lowpass
highpass = generate_signal2(cutoff_freq=5000, length=1000)  # 5 kHz highpass

# Generate mixed response
lp, hp, mixed = generate_sample(fc1=1000, fc2=5000, length=1000)
```

### Random Dataset Generation
```python
from data.generation import generate_random_dataset

# Generate dataset with random cutoff frequencies
lowpass_data, highpass_data, mixed_data = generate_random_dataset(
    N=1000, 
    length=512,
    fc1_range=(10, 10000),    # Lowpass: 10 Hz to 10 kHz
    fc2_range=(100, 100000),  # Highpass: 100 Hz to 100 kHz
    seed=42
)
```

### Advanced Filter Types
```python
from data.generation import generate_bandpass_filter, generate_notch_filter

# Bandpass filter (1 kHz to 10 kHz)
bandpass = generate_bandpass_filter(fc_low=1000, fc_high=10000, length=1000)

# Notch filter at 5 kHz with Q=10
notch = generate_notch_filter(fc_center=5000, Q=10, length=1000)
```

## Model Compatibility

The transformer model remains unchanged as it operates on the signal level. The key benefits of this approach:

1. **Realistic Training Data**: Filter responses represent actual electronic circuit behavior
2. **Frequency Domain Focus**: Natural log-scale frequency distribution matches real-world scenarios
3. **Circuit Design Applications**: Decomposed responses can guide analog circuit synthesis
4. **Scalable Complexity**: Easy to add more filter types (bandpass, notch, etc.)

## Testing

Run the test script to visualize the new filter-based generation:
```bash
python test_filter_generation.py
```

This will generate plots showing:
- Individual lowpass and highpass responses
- Mixed responses 
- Component decomposition
- Bandpass and notch filter examples
- Phase response examples
- Random dataset samples
