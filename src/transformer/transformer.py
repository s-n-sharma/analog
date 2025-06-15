import torch
import torch.nn as nn

class CircuitTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(CircuitTransformer, self).__init__()  
        self.d_model = d_model