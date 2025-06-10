import torch.nn as nn
import torch
import numpy as np
import math
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model : int, n_heads : int, dropout=0.0, bias=True, flash=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        
        # linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        # output 
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.d_head ** -0.5   
        self.flash = flash 
    
    def _split_heads(self, x : torch.Tensor):
        B, T, _ = x.size()
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
    def _combine_heads(self, x : torch.Tensor):
        B, H, T, d_h = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
    def forward(self, x, mask=None):
        Q = self._split_heads(self.W_q(x))      
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        
        scores = (Q @ K.transpose(-2, -1)) * self.scale   
        if mask is not None:                              
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(scores.softmax(dim=-1))
        out  = attn @ V                                 

        out = self._combine_heads(out)              
        return self.W_o(out)
class MultiHeadCrossAttention(nn.Module):
    """
    Query = decoder hidden state (x)
    Key / Value = encoder memory (context)
    Shapes
        x        : [B, T_dec, d_model]
        context  : [B, T_enc, d_model]
        mask     : [B, 1, T_dec, T_enc]   (optional; 0 = block)
    """
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        # Q comes from decoder; K/V from encoder
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    # — helpers (same as the self‑attention class) —
    def _split_heads(self, t: torch.Tensor):
        B, T, _ = t.size()
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,d_h]

    def _combine_heads(self, t: torch.Tensor):
        B, H, T, d_h = t.size()
        return t.transpose(1, 2).contiguous().view(B, T, self.d_model)

    # — forward —
    def forward(self, x: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor | None = None):
        Q = self._split_heads(self.W_q(x))          # [B,H,T_dec,d_h]
        K = self._split_heads(self.W_k(context))    # [B,H,T_enc,d_h]
        V = self._split_heads(self.W_v(context))

        scores = (Q @ K.transpose(-2, -1)) * self.scale   # [B,H,T_dec,T_enc]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.dropout(scores.softmax(dim=-1))
        out  = attn @ V                             

        out = self._combine_heads(out)                 
        return self.W_o(out)
    
    
       
class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fn      = fn                      # store sub‑layer
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.fn(self.norm(x), *args, **kwargs)
        return x + self.dropout(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 dropout: float = 0.1, activation: str = "gelu",
                 gated: bool = False):
        super().__init__()                             # ← missing

        d_ff = d_ff or 4 * d_model
        self.gated = gated

        self.lin1 = nn.Linear(d_model, d_ff * (2 if gated else 1))
        self.lin2 = nn.Linear(d_ff, d_model)           # output dim fixed
        self.dropout = nn.Dropout(dropout)

        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        if self.gated:                                
            u, v = self.lin1(x).chunk(2, dim=-1)
            x = self.lin2(self.dropout(self.act(u) * v))
        else:
            x = self.lin2(self.dropout(self.act(self.lin1(x))))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn  = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.resid_attn = ResidualBlock(self.attn, d_model, dropout)
        self.resid_ffn  = ResidualBlock(self.ffn,  d_model, dropout)

    def forward(self, x, mask=None):
        x = self.resid_attn(x, mask=mask)  
        x = self.resid_ffn(x)              
        return x



class Decoder(nn.Module):

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()

        self.self_attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model,
                                                  d_ff or 4 * d_model,
                                                  dropout)

   
        self.resid_self  = ResidualBlock(self.self_attn,  d_model, dropout)
        self.resid_cross = ResidualBlock(self.cross_attn, d_model, dropout)
        self.resid_ffn   = ResidualBlock(self.ffn,        d_model, dropout)


    def forward(
        self,
        x: torch.Tensor,                   
        memory: torch.Tensor,             
        tgt_mask: torch.Tensor | None = None,     
        memory_mask: torch.Tensor | None = None, 
    ) -> torch.Tensor:
        # masked self attention
        x = self.resid_self(x, mask=tgt_mask)

        # cross attention
        x = self.resid_cross(x, context=memory, mask=memory_mask)

        # feed forward
        x = self.resid_ffn(x)
        return x