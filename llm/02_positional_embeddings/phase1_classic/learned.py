"""
Learned Positional Embeddings

Trainable positional embeddings as used in BERT, GPT-2, and GPT-3.
Each position has a learnable embedding vector that is optimized during training.

Key properties:
- Learnable parameters (max_len × d_model)
- Can be optimized for specific tasks
- Limited to max_len (cannot extrapolate)
- Used in most pre-transformer era models
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    create_extended_vocabulary,
    DATA_PROCESSED_DIR,
    print_section
)


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Simple embedding lookup table for positions.
    """

    def __init__(self, max_len: int, d_model: int):
        """
        Initialize learned positional embedding.

        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
        """
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        # Learnable embedding table
        self.embedding = nn.Embedding(max_len, d_model)

        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embedding to input.

        Args:
            x: Input embeddings (batch, seq_len, d_model)

        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        assert seq_len <= self.max_len, f"Sequence length {seq_len} exceeds max_len {self.max_len}"

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Look up positional embeddings
        pos_embeddings = self.embedding(positions)

        return x + pos_embeddings

    def get_encoding(self, max_pos: int) -> torch.Tensor:
        """Get encoding for positions [0, max_pos)."""
        positions = torch.arange(max_pos, device=self.embedding.weight.device)
        return self.embedding(positions)


class SimpleLanguageModelDataset(Dataset):
    """Simple dataset for training learned positional embeddings."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        """
        Create synthetic language modeling data.

        Args:
            vocab_size: Size of vocabulary
            seq_len: Sequence length
            num_samples: Number of training samples
        """
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        # In real training, these would be actual text
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        return self.data[idx, :-1], self.data[idx, 1:]


def train_learned_embeddings(
    vocab_size: int = 100,
    d_model: int = 64,
    max_len: int = 128,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001
):
    """Train learned positional embeddings with a simple language model."""
    print_section("Training Learned Positional Embeddings")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    print(f"Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  max_len: {max_len}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}\n")

    # Create model components
    token_embedding = nn.Embedding(vocab_size, d_model)
    pos_embedding = LearnedPositionalEmbedding(max_len, d_model)
    output_projection = nn.Linear(d_model, vocab_size)

    # Move to device
    token_embedding = token_embedding.to(device)
    pos_embedding = pos_embedding.to(device)
    output_projection = output_projection.to(device)

    # Create dataset
    dataset = SimpleLanguageModelDataset(vocab_size, seq_len=12, num_samples=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    all_params = list(token_embedding.parameters()) + \
                 list(pos_embedding.parameters()) + \
                 list(output_projection.parameters())
    optimizer = optim.Adam(all_params, lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Count parameters
    pos_params = sum(p.numel() for p in pos_embedding.parameters())
    total_params = sum(p.numel() for p in all_params)
    print(f"Model parameters:")
    print(f"  Positional embedding: {pos_params:,}")
    print(f"  Total: {total_params:,}\n")

    print("Training...")
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            token_emb = token_embedding(inputs)  # (batch, seq_len, d_model)
            embeddings = pos_embedding(token_emb)  # Add positional embeddings
            logits = output_projection(embeddings)  # (batch, seq_len, vocab_size)

            # Compute loss
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    print("\nTraining complete!")

    # Save learned embeddings
    save_path = DATA_PROCESSED_DIR / "learned_embeddings.pt"
    torch.save({
        'encodings': pos_embedding.get_encoding(max_len).cpu(),
        'state_dict': pos_embedding.state_dict(),
        'd_model': d_model,
        'max_len': max_len
    }, save_path)
    print(f"Saved learned embeddings to: {save_path}")

    # Get encodings for analysis
    encodings = pos_embedding.get_encoding(20).cpu().detach().numpy()

    # Show properties
    print(f"\nLearned Positional Embedding Properties:")
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

    return pos_embedding, encodings


def main():
    parser = argparse.ArgumentParser(description="Train Learned Positional Embeddings")
    parser.add_argument("--vocab-size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_learned_embeddings(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        max_len=args.max_len,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
