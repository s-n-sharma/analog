import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    A basic Transformer model for sequence-to-sequence tasks.
    """
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, d_hid, dropout)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            tgt: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
            tgt_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output: Tensor, shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.decoder(output)
        return output

if __name__ == '__main__':
    # --- Hyperparameters ---
    ntokens = 1000  # Size of vocabulary
    d_model = 200  # Embedding dimension
    d_hid = 200  # Dimension of the feedforward network model
    nlayers = 2  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # Number of heads in nn.MultiheadAttention
    dropout = 0.2  # Dropout value

    model = TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout)

    # --- Dummy Data ---
    batch_size = 10
    seq_len = 15
    src = torch.randint(0, ntokens, (seq_len, batch_size))  # (S, N)
    tgt = torch.randint(0, ntokens, (seq_len, batch_size))  # (T, N)

    # --- Masks ---
    # The source mask is used to prevent the model from attending to padding tokens.
    # For this basic example, we will use a square attention mask.
    src_mask = torch.zeros((seq_len, seq_len))

    # The target mask is a causal mask to prevent the model from attending to future tokens.
    tgt_mask = model.transformer.generate_square_subsequent_mask(seq_len)

    # --- Forward Pass ---
    output = model(src, tgt, src_mask, tgt_mask)

    print("Input source shape:", src.shape)
    print("Input target shape:", tgt.shape)
    print("Output shape:", output.shape)