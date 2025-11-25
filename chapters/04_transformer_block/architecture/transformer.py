import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from architecture.ffn import FeedForward
from architecture.multi_head import MultiHeadAttention
from constants import D_MODEL, DROPOUT


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    """
    def __init__(self, apply_rope: bool = False) -> None:
        """
        Initialize the transformer block.
        """
        super().__init__()
        self.attention = MultiHeadAttention(apply_rope=apply_rope)
        self.feed_forward = FeedForward(D_MODEL)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))

        # Feed-forward with residual
        x = x + self.feed_forward(self.norm2(x))
        return x