import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import DROPOUT


class FeedForward(nn.Module):
    """
    Feed-forward network for the transformer.
    """
    def __init__(self, d_model: int) -> None:
        """
        Initialize the feed-forward network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.
        """
        return self.net(x)