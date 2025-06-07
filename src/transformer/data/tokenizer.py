import numpy as np 
import torch.nn as nn
import re
import os # For folder processing
from collections import Counter

"""This is the input tokenizer for the transformer model. It converts the Bode plot data into a tokenized sequence."""
class BodeTokenizer:
    def __init__(self, 
                 magnitude_bins=100,  # Number of bins for magnitude discretization
                 phase_bins=100,      # Number of bins for phase discretization
                 max_freq_points=100, 
                 magnitude_range=(-60, 20),  
                 phase_range=(-180, 180), only_magnitude=True):   
        
        self.magnitude_bins = magnitude_bins
        self.phase_bins = phase_bins
        self.max_freq_points = max_freq_points
        self.only_magnitude = only_magnitude
        
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
        if self.only_magnitude:
            # Magnitude only mode - do not include phase data
            if len(frequencies) > self.max_freq_points:
                indices = np.linspace(0, len(frequencies)-1, self.max_freq_points, dtype=int)
                frequencies = frequencies[indices]
                magnitudes = magnitudes[indices]
            mag_tokens = self._discretize(magnitudes, self.magnitude_edges) + 5  # Offset by special tokens
            tokens = [self.SOS_TOKEN, self.MAG_TOKEN]
            tokens.extend(mag_tokens.tolist())
            tokens.append(self.EOS_TOKEN)
            return tokens
            
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
    def __init__(self, special_tokens_override=None):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # Define standard special tokens
        self.PAD_TOKEN_STR = "<PAD>"
        self.SOS_TOKEN_STR = "<SOS>"
        self.EOS_TOKEN_STR = "<EOS>"
        self.UNK_TOKEN_STR = "<UNK>"
        self.NEWLINE_TOKEN_STR = "<NEWLINE>"

        default_special_tokens = [
            self.PAD_TOKEN_STR, self.SOS_TOKEN_STR, self.EOS_TOKEN_STR, 
            self.UNK_TOKEN_STR, self.NEWLINE_TOKEN_STR
        ]
        
        self.special_tokens = special_tokens_override if special_tokens_override is not None else default_special_tokens
        
        self._init_vocab_with_special_tokens()

    def _init_vocab_with_special_tokens(self):
        """Initializes/resets vocabulary with special tokens, ensuring PAD is 0."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Ensure PAD_TOKEN is 0 if it's one of the special tokens
        if self.PAD_TOKEN_STR in self.special_tokens:
            self.token_to_id[self.PAD_TOKEN_STR] = 0
            self.id_to_token[0] = self.PAD_TOKEN_STR
        
        current_id = len(self.token_to_id)
        for token in self.special_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        self.vocab_size = len(self.token_to_id)

    def _tokenize_single_line(self, line):
        line = line.strip()
        if not line or line.startswith('//'): 
            return []

        parts = re.split(r'\s+', line) 
        tokens = []

        if not parts:
            return []

        command = parts[0]
        tokens.append(command)

        if command in ["Vin", "Vout", ".end"]:
            tokens.extend(parts[1:])
        elif command == ".fn":
            if len(parts) >= 4: # .fn node1 node2 func_name ...params
                tokens.extend(parts[1:4]) # node1, node2, func_name
                for param_assignment in parts[4:]:
                    if '=' in param_assignment:
                        name, value = param_assignment.split('=', 1)
                        tokens.append(name)
                        tokens.append('=')
                        tokens.append(value)
                    else:
                        tokens.append(param_assignment) 
            else:
                pass 
        else:
            tokens.extend(parts[1:])
            
        return tokens

    def _get_string_tokens_from_script(self, script_content):
        """Converts a full netlist script string into a list of raw string tokens, including SOS/EOS/NEWLINE."""
        string_tokens = [self.SOS_TOKEN_STR]
        lines = script_content.strip().split('\n')
        
        for line_str in lines:
            line_s_tokens = self._tokenize_single_line(line_str)
            if line_s_tokens:
                string_tokens.extend(line_s_tokens)
                string_tokens.append(self.NEWLINE_TOKEN_STR)
        if len(string_tokens) > 1 and \
           string_tokens[-1] == self.NEWLINE_TOKEN_STR and \
           string_tokens[-2] == ".end":
            string_tokens[-1] = self.EOS_TOKEN_STR
        # If the sequence just ends with ".end" (e.g. from a single line script ".end"), append <EOS>
        elif string_tokens[-1] == ".end":
            string_tokens.append(self.EOS_TOKEN_STR)
        # Otherwise, if it doesn't end with EOS, append EOS.
        # This also handles removing a trailing NEWLINE if it's not part of an ".end<EOS>" sequence.
        elif string_tokens[-1] != self.EOS_TOKEN_STR:
            if string_tokens[-1] == self.NEWLINE_TOKEN_STR:
                string_tokens.pop() # Remove trailing NEWLINE
            if not string_tokens or string_tokens[-1] != self.EOS_TOKEN_STR: # Check again after pop
                 string_tokens.append(self.EOS_TOKEN_STR)
        
        # Handle empty script case: [<SOS>, <EOS>]
        if len(string_tokens) == 1 and string_tokens[0] == self.SOS_TOKEN_STR: # Only SOS means empty script
            string_tokens.append(self.EOS_TOKEN_STR)
            
        return string_tokens

    def build_vocab_from_folder(self, folder_path, file_extension=".sp"):
        """
        Processes all circuit files in a folder to build/update the vocabulary.
        """
        print(f"Building vocabulary from files in: {folder_path}")
        self._init_vocab_with_special_tokens() # Reset and start with special tokens

        token_counts = Counter()
        filepaths = []
        for item in os.listdir(folder_path):
            if item.endswith(file_extension):
                filepaths.append(os.path.join(folder_path, item))

        for filepath in filepaths:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Get string tokens, but exclude special tokens for counting purposes here,
                # as they are already handled.
                script_s_tokens = self._get_string_tokens_from_script(content)
                # Filter out pre-defined special tokens before counting, to avoid re-adding them.
                # We only want to count and add corpus-specific tokens.
                corpus_tokens = [t for t in script_s_tokens if t not in self.special_tokens]
                token_counts.update(corpus_tokens)
            except Exception as e:
                print(f"Warning: Could not process file {filepath} for vocab: {e}")
                continue
        
        # Add corpus tokens to vocabulary
        current_id = self.vocab_size # Start adding after special tokens
        for token, _ in token_counts.most_common(): # Iterate by frequency
            if token not in self.token_to_id: # Add only if not already a special token
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
        
        self.vocab_size = len(self.token_to_id)
        print(f"Vocabulary built. Total size: {self.vocab_size} tokens.")
        if self.vocab_size == len(self.special_tokens) and token_counts:
            print("Warning: No new tokens found in corpus beyond special tokens. Check files/extensions.")


    def tokenizes(self, script_content):
        """Converts a netlist script string into a sequence of integer token IDs."""
        if not self.token_to_id or self.vocab_size <= len(self.special_tokens):
            # Check if vocab seems uninitialized beyond special tokens
            print("Warning: Vocabulary might not be fully built or is empty beyond special tokens. Call build_vocab_from_folder() on your dataset first.")
            if not self.token_to_id.get(self.UNK_TOKEN_STR): # Critical if UNK is missing
                 raise ValueError("UNK_TOKEN not in vocabulary. Vocabulary is not correctly initialized.")


        string_tokens = self._get_string_tokens_from_script(script_content)
        
        unk_token_id = self.token_to_id.get(self.UNK_TOKEN_STR)
        # This check is vital. If UNK_TOKEN_STR itself wasn't added, something is wrong.
        if unk_token_id is None:
             # This should ideally be caught by _init_vocab_with_special_tokens
             raise ValueError(f"Critical error: UNK_TOKEN_STR ('{self.UNK_TOKEN_STR}') not found in token_to_id map.")

        id_sequence = [self.token_to_id.get(token, unk_token_id) for token in string_tokens]
        return id_sequence

    def decode(self, id_sequence):
        """Converts a sequence of integer token IDs back into a list of string tokens."""
        unk_token_str = self.UNK_TOKEN_STR 
        return [self.id_to_token.get(id_val, unk_token_str) for id_val in id_sequence]

    def process_folder_to_token_ids(self, folder_path, file_extension=".sp", build_vocab_if_empty=True):
        """
        Processes all circuit files in a folder. Builds vocabulary if it seems empty,
        then converts all files to token ID sequences.
        
        Returns:
            A list of tuples: (filepath, id_sequence)
        """
        if build_vocab_if_empty and (not self.token_to_id or self.vocab_size <= len(self.special_tokens)):
            print("Vocabulary seems uninitialized or only contains special tokens. Building from folder first.")
            self.build_vocab_from_folder(folder_path, file_extension)
            if self.vocab_size <= len(self.special_tokens) and os.listdir(folder_path):
                 print("Warning: Vocabulary building did not add new tokens. Ensure files exist and are readable.")


        if not self.token_to_id.get(self.UNK_TOKEN_STR):
            raise ValueError("Tokenizer not properly initialized or vocabulary not built (UNK_TOKEN missing).")

        results = []
        for filename in os.listdir(folder_path):
            if filename.endswith(file_extension):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    id_sequence = self.tokenizes(content)
                    results.append((filepath, id_sequence))
                except Exception as e:
                    print(f"Warning: Could not tokenize file {filepath}: {e}")
                    continue
        
        return results
