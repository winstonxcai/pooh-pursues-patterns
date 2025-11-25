import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import D_MODEL, DEVICE, DROPOUT, NUM_HEADS, SEQ_LEN


class SingleHeadAttention(nn.Module):
    """
    Single-head attention for the transformer.
    """
    def __init__(self, apply_rope: bool = False) -> None:
        """
        Initialize the single-head attention.
        """
        super().__init__()
        head_size = D_MODEL // NUM_HEADS
        self.q = nn.Linear(D_MODEL, head_size, bias=False)
        self.k = nn.Linear(D_MODEL, head_size, bias=False)
        self.v = nn.Linear(D_MODEL, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(SEQ_LEN, SEQ_LEN)))
        self.dropout = nn.Dropout(DROPOUT)
        self.apply_rope = apply_rope

    def rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        apply RoPE to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        half = d_model // 2

        # (seq_len,)
        positions = torch.arange(seq_len, device=DEVICE)

        # (seq_len, half)
        theta = 1.0 / (10000.0 ** (torch.arange(half, device=DEVICE) / half))

        # (seq_len, half)
        # outer treats A as rows and B as columns
        freqs = torch.outer(positions, theta) # (seq_len, half)
        freqs = freqs.unsqueeze(0) # (1, seq_len, half)

        # setup for creating x_i and x_i+1 pairs
        odd = x[..., 0::2] # (batch_size, seq_len, half)
        even = x[..., 1::2] # (batch_size, seq_len, half)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        x_rot_even = (even * cos) + (odd * sin)
        x_rot_odd = (odd * cos) - (even * sin)

        # concatenate into dimensions of (batch_size, seq_len, d_model, 2)  NOTE: it does not interleave, that is done in the last operation with the reshaping
        x_out = torch.stack([x_rot_odd, x_rot_even], dim=-1)

        #rehapsing merges the final two dimensions, of the tensor into one dimension
        # so that the output tensor has shape (batch_size, seq_len, d_model)
        return x_out.reshape(*x.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply single-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, head_size)
        """
        batch_size, seq_len, d_model = x.shape
        # Linear layers replace the final dimension since (batch_size, seq_len, d_model) @ (d_model, head_size) applies
        Q = self.q(x)  # (batch_size, seq_len, head_size) 
        K = self.k(x)  # (batch_size, seq_len, head_size)
        V = self.v(x)  # (batch_size, seq_len, head_size)

        if self.apply_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        # Q @ K^T / (head_size^0.5)
        scaled_dot_product = torch.matmul(Q, K.transpose(-2, -1)) * K.shape[-1]**-0.5 # (batch_size, seq_len, seq_len)

        # casual attention mask
        masked = scaled_dot_product.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf")) # (batch_size, seq_len, seq_len)

        # apply softmax
        soft = F.softmax(masked, dim=-1)

        # apply dropout
        dropped = self.dropout(soft)

        # Map to the values: (batch_size, seq_len, seq_len) * (batch_size, seq_len, head_size)
        output = torch.matmul(dropped, V)
        return output



