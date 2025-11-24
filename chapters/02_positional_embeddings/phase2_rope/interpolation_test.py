"""
RoPE Interpolation and Extrapolation Tests

Tests RoPE's ability to generalize to sequence lengths beyond training:
- Train on short sequences (e.g., length 6)
- Test on longer sequences (e.g., length 12, 24, 48)
- Compare with sinusoidal and learned embeddings
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    setup_plotting_style,
    save_figure,
    print_section
)
from shared.toy_model import ToyTransformer


class SimpleSeqDataset(Dataset):
    """Simple sequence dataset for testing extrapolation."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 500):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


def train_model(model, dataloader, device, epochs=5, lr=0.001):
    """Train a simple language model."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    return model


def evaluate_model(model, vocab_size, seq_len, device, num_samples=100):
    """Evaluate model on sequences of given length."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Create test dataset
    test_dataset = SimpleSeqDataset(vocab_size, seq_len, num_samples)
    test_loader = DataLoader(test_dataset, batch_size=32)

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            try:
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item()
                num_batches += 1
            except (RuntimeError, AssertionError) as e:
                # Model can't handle this length (e.g., learned embeddings)
                return float('inf')

    return total_loss / num_batches if num_batches > 0 else float('inf')


def test_extrapolation():
    """Test extrapolation capabilities of different position encoding methods."""
    print_section("Testing Length Extrapolation")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Parameters
    vocab_size = 50
    d_model = 64
    n_heads = 4
    train_seq_len = 6
    test_seq_lens = [6, 12, 24, 48]

    print(f"Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  train_seq_len: {train_seq_len}")
    print(f"  test_seq_lens: {test_seq_lens}\n")

    # Methods to test
    methods = ["sinusoidal", "learned", "rope", "alibi"]
    results = {method: [] for method in methods}

    for method in methods:
        print(f"\nTesting {method.upper()}...")
        print("-" * 80)

        # Create model
        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=2,
            d_ff=256,
            max_seq_len=max(test_seq_lens),  # Support longest test length
            position_encoding=method
        ).to(device)

        # Create training dataset
        train_dataset = SimpleSeqDataset(vocab_size, train_seq_len, num_samples=500)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train
        print(f"Training on sequences of length {train_seq_len}...")
        model = train_model(model, train_loader, device, epochs=3)

        # Evaluate on different lengths
        print(f"\nEvaluating on different sequence lengths...")
        for test_len in test_seq_lens:
            loss = evaluate_model(model, vocab_size, test_len, device)
            results[method].append(loss)

            if loss == float('inf'):
                print(f"  Length {test_len:2d}: FAILED (model cannot handle this length)")
            else:
                print(f"  Length {test_len:2d}: Loss = {loss:.4f}")

    # Plot results
    print("\n" + "=" * 80)
    print("Plotting extrapolation results...")

    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'sinusoidal': 'blue', 'learned': 'red', 'rope': 'green', 'alibi': 'purple'}
    markers = {'sinusoidal': 'o', 'learned': 's', 'rope': '^', 'alibi': 'D'}

    for method in methods:
        losses = results[method]
        # Replace inf with None for plotting
        losses_plot = [l if l != float('inf') else None for l in losses]

        ax.plot(test_seq_lens, losses_plot,
               marker=markers[method], linewidth=2, markersize=8,
               label=method.upper(), color=colors[method])

    ax.axvline(x=train_seq_len, color='black', linestyle='--', linewidth=1.5,
              label=f'Train length ({train_seq_len})', alpha=0.5)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Test Loss (lower is better)')
    ax.set_title('Length Extrapolation Test\n(Train on length 6, test on longer sequences)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    save_figure(fig, "rope_extrapolation_comparison.png", subdir='phase2')
    plt.close()

    # Print summary table
    print("\n" + "=" * 80)
    print("EXTRAPOLATION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<15} ", end='')
    for length in test_seq_lens:
        print(f"{'Len ' + str(length):>12}", end='')
    print()
    print("-" * 80)

    for method in methods:
        print(f"{method.upper():<15} ", end='')
        for loss in results[method]:
            if loss == float('inf'):
                print(f"{'FAILED':>12}", end='')
            else:
                print(f"{loss:>12.4f}", end='')
        print()

    print("=" * 80)

    # Analysis
    print("\nKEY FINDINGS:")
    print("-" * 80)

    # Check which methods handle all lengths
    for method in methods:
        all_finite = all(l != float('inf') for l in results[method])
        if all_finite:
            # Compute degradation
            train_loss = results[method][0]  # Loss at train length
            max_loss = max(results[method])
            degradation = ((max_loss - train_loss) / train_loss) * 100

            print(f"\n{method.upper()}:")
            print(f"  ✓ Handles all test lengths")
            print(f"  ✓ Train length loss: {train_loss:.4f}")
            print(f"  ✓ Max length loss: {max_loss:.4f}")
            print(f"  ✓ Degradation: {degradation:.1f}%")
        else:
            print(f"\n{method.upper()}:")
            print(f"  ✗ Cannot extrapolate beyond trained length")

    print("\n" + "=" * 80)
    print("Extrapolation test complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test RoPE Extrapolation")
    args = parser.parse_args()

    test_extrapolation()


if __name__ == "__main__":
    main()
