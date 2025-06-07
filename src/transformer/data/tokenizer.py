import numpy as np 
import torch.nn as nn


"""This is the input tokenizer for the transformer model. It converts the Bode plot data into a tokenized sequence."""
class BodeTokenizer:
    def __init__(self, 
                 magnitude_bins=100,  # Number of bins for magnitude discretization
                 phase_bins=100,      # Number of bins for phase discretization
                 max_freq_points=100, 
                 magnitude_range=(-60, 20),  
                 phase_range=(-180, 180)):   
        
        self.magnitude_bins = magnitude_bins
        self.phase_bins = phase_bins
        self.max_freq_points = max_freq_points
        
        # Create bin edges for discretization
        self.magnitude_edges = np.linspace(magnitude_range[0], magnitude_range[1], magnitude_bins + 1)
        self.phase_edges = np.linspace(phase_range[0], phase_range[1], phase_bins + 1)
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.SOS_TOKEN = 1  # Start of sequence
        self.EOS_TOKEN = 2  # End of sequence
        self.MAG_TOKEN = 3  # Start of magnitude data
        self.PHASE_TOKEN = 4  # Start of phase data
        
        # Calculate vocabulary size
        self.vocab_size = magnitude_bins + phase_bins + 5  # +5 for special tokens

    def _discretize(self, values, edges):
        """Convert continuous values to discrete bins."""
        return np.digitize(values, edges) - 1

    def tokenize(self, frequencies, magnitudes, phases):
        """
        Convert Bode plot data into a tokenized sequence.
        
        Args:
            frequencies: Array of frequency points
            magnitudes: Array of magnitude values in dB
            phases: Array of phase values in degrees
            
        Returns:
            tokens: List of token IDs representing the Bode plot
        """
    
        if len(frequencies) > self.max_freq_points:
            indices = np.linspace(0, len(frequencies)-1, self.max_freq_points, dtype=int)
            frequencies = frequencies[indices]
            magnitudes = magnitudes[indices]
            phases = phases[indices]
        
        # Discretize magnitude and phase data
        mag_tokens = self._discretize(magnitudes, self.magnitude_edges) + 5  # Offset by special tokens
        phase_tokens = self._discretize(phases, self.phase_edges) + 5 + self.magnitude_bins
        
        # Create sequence with special tokens
        tokens = [self.SOS_TOKEN]
        tokens.append(self.MAG_TOKEN)
        tokens.extend(mag_tokens.tolist())
        tokens.append(self.PHASE_TOKEN)
        tokens.extend(phase_tokens.tolist())
        tokens.append(self.EOS_TOKEN)
        
        return tokens
    
"""This is the output data tokenizer, it converts a circuit netlist into a tokenized sequence."""
class CircuitTokenizer:
    def __init__(self):
        pass
    
    def tokenize(self, netlist):
        pass

