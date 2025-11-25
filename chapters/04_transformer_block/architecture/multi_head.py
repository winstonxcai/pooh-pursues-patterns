import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from architecture.single_head import SingleHeadAttention
from constants import D_MODEL, DROPOUT, NUM_HEADS


class MultiHeadAttention(nn.Module):
    def __init__(self, apply_rope: bool = False) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(apply_rope=apply_rope) for _ in range(NUM_HEADS)])
        self.W_out = nn.Linear(D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        concatenation = torch.cat([h(x) for h in self.heads], dim=-1)

        # (batch_size, seq_len, d_model) @ (d_model, d_model)
        final_projection = self.W_out(concatenation)

        output = self.dropout(final_projection)
        return output