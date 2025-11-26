"""
Sinusoidal Positional Embeddings (Vaswani et al., 2017)

The original positional encoding from "Attention Is All You Need".
Uses sine and cosine functions at different frequencies to encode position.

Key properties:
- Deterministic (no learned parameters)
- Each position gets a unique encoding
- Can generalize to any sequence length
- Linear relationships between positions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    DATA_PROCESSED_DIR,
    print_section
)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Embedding dimension (must be even)
            max_len: Maximum sequence length
        """
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute div_term: 10000^(2i/d_model) for i in [0, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(np.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

    def get_encoding(self, max_pos: int) -> torch.Tensor:
        """Get encoding for positions [0, max_pos)."""
        return self.pe[:max_pos, :]


def demonstrate_sinusoidal():
    """Demonstrate sinusoidal positional encoding."""
    print_section("Sinusoidal Positional Encoding Demo")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Parameters
    d_model = 64
    max_len = 128

    # Create sinusoidal encoding
    pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
    pos_encoder = pos_encoder.to(device)

    print(f"Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    print(f"  Parameters: 0 (fixed, not learned)\n")

    # Load toy sequence
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}")
    print(f"Vocabulary size: {len(vocab)}\n")

    # Convert to IDs
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)
    seq_len = token_ids.size(1)

    # Create dummy token embeddings
    token_emb = torch.randn(1, seq_len, d_model, device=device)

    # Add positional encoding
    embeddings_with_pos = pos_encoder(token_emb)

    print(f"Results:")
    print(f"  Input shape: {token_emb.shape}")
    print(f"  Output shape: {embeddings_with_pos.shape}")
    print(f"  Positional encoding shape: {pos_encoder.pe.shape}\n")

    # Get encodings for visualization
    encodings = pos_encoder.get_encoding(20).cpu().detach().numpy()

    # Show properties
    print(f"Positional Encoding Properties:")
    print(f"  Position 0 encoding (first 8 dims): {encodings[0, :8]}")
    print(f"  Position 1 encoding (first 8 dims): {encodings[1, :8]}")
    print(f"  Position 5 encoding (first 8 dims): {encodings[5, :8]}")
    print()

    # Compute similarities between positions
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(encodings[:10])

    print(f"Cosine Similarity Between Positions:")
    print(f"  Pos 0 vs Pos 1: {similarities[0, 1]:.4f}")
    print(f"  Pos 0 vs Pos 5: {similarities[0, 5]:.4f}")
    print(f"  Pos 1 vs Pos 2: {similarities[1, 2]:.4f}")
    print(f"  Pos 5 vs Pos 6: {similarities[5, 6]:.4f}")
    print()

    # Save encodings for visualization
    save_path = DATA_PROCESSED_DIR / "sinusoidal_encodings.pt"
    torch.save({
        'encodings': pos_encoder.pe.cpu(),
        'd_model': d_model,
        'max_len': max_len
    }, save_path)
    print(f"Saved encodings to: {save_path}")

    # Demonstrate frequency spectrum
    print(f"\nFrequency Spectrum (wavelengths across dimensions):")
    for i in range(0, d_model, 8):
        div_term = np.exp(i * -(np.log(10000.0) / d_model))
        wavelength = 2 * np.pi / div_term
        print(f"  Dim {i:2d}: wavelength = {wavelength:8.2f} positions")

    return pos_encoder, encodings


def main():
    parser = argparse.ArgumentParser(description="Sinusoidal Positional Encoding Demo")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length")
    args = parser.parse_args()

    demonstrate_sinusoidal()


if __name__ == "__main__":
    main()
