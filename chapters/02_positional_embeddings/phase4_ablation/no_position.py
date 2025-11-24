"""
Ablation Study: No Positional Encoding

Demonstrates what happens when positional information is removed entirely.

This shows:
- Why transformers need position information
- How attention becomes position-agnostic
- Catastrophic failure on position-dependent tasks
- Permutation invariance
"""

import argparse
import sys
from pathlib import Path

import numpy as np
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
    load_toy_sequence,
    tokens_to_ids,
    DATA_PROCESSED_DIR,
    print_section
)
from shared.toy_model import ToyTransformer


def demonstrate_no_position():
    """Demonstrate transformer without positional encoding."""
    print_section("Transformer WITHOUT Positional Encoding")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Load toy sequence
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}")
    print(f"Tokens: {tokens}\n")

    # Convert to IDs
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)

    # Create model WITHOUT position encoding
    vocab_size = len(vocab)
    model = ToyTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=128,
        position_encoding="none"  # No positional encoding!
    ).to(device)

    print("Model configuration:")
    print(f"  Position encoding: NONE")
    print(f"  d_model: 64")
    print(f"  n_heads: 4")
    print(f"  n_layers: 2\n")

    # Get embeddings
    embeddings = model.get_embeddings(token_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"  (batch, seq_len, d_model) = (1, {len(tokens)}, 64)\n")

    # Forward pass
    logits, attention_weights = model(token_ids, return_attention=True)

    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape (per layer): {attention_weights[0].shape}\n")

    # Show attention pattern
    print("Attention pattern (Layer 0, Head 0):")
    attn = attention_weights[0][0, 0].cpu().detach().numpy()
    print("  (rows=query, cols=key)")
    print(attn)
    print()

    # Test permutation invariance
    print("=" * 80)
    print("TESTING PERMUTATION INVARIANCE")
    print("=" * 80)
    print()

    # Original sequence
    print(f"Original sequence: {' '.join(tokens)}")
    output_original = model.get_embeddings(token_ids)[0]

    # Permuted sequences
    permutations = [
        ["cat", "the", "on", "sat", "mat", "the"],
        ["mat", "the", "sat", "on", "cat", "the"],
        ["the", "the", "cat", "mat", "on", "sat"],
    ]

    print("\nComparing outputs for different permutations:")
    print()

    for perm in permutations:
        perm_ids = torch.tensor(
            [vocab[t] for t in perm], dtype=torch.long, device=device
        ).unsqueeze(0)

        output_perm = model.get_embeddings(perm_ids)[0]

        # Compute difference
        diff = torch.norm(output_original - output_perm).item()

        print(f"Permutation: {' '.join(perm)}")
        print(f"  Difference from original: {diff:.6f}")

        if diff < 1e-5:
            print(f"  → IDENTICAL (permutation invariance confirmed!)")
        else:
            print(f"  → Different")
        print()

    # Show why this is a problem
    print("=" * 80)
    print("WHY THIS IS A PROBLEM")
    print("=" * 80)
    print()

    print("Without position information:")
    print()
    print("  1. The model cannot distinguish between:")
    print("     'the cat sat on the mat' and 'the mat sat on the cat'")
    print()
    print("  2. Word order is meaningless:")
    print("     'cat the sat' = 'the cat sat' = 'sat cat the'")
    print()
    print("  3. Tokens at different positions have identical representations")
    print("     (if they're the same word)")
    print()
    print("  4. Attention is position-agnostic:")
    print("     Cannot attend to 'next token' or 'previous token'")
    print()

    # Save results
    save_path = DATA_PROCESSED_DIR / "no_position_results.pt"
    torch.save({
        'attention_weights': [aw.cpu() for aw in attention_weights],
        'embeddings': embeddings.cpu(),
        'tokens': tokens
    }, save_path)
    print(f"Saved results to: {save_path}")

    return model, attention_weights


class SequenceDataset(Dataset):
    """Dataset for sequence modeling task."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 500):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.data[idx, 1:]


def compare_with_vs_without_position():
    """Compare model performance with and without positional encoding."""
    print_section("Comparing WITH vs WITHOUT Position")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    vocab_size = 50
    d_model = 64
    seq_len = 12

    print(f"Configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  Training samples: 500")
    print(f"  Epochs: 5\n")

    # Create dataset
    dataset = SequenceDataset(vocab_size, seq_len, num_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train both models
    methods = {
        "With Position (Sinusoidal)": "sinusoidal",
        "WITHOUT Position": "none"
    }

    results = {}

    for name, pos_encoding in methods.items():
        print(f"\n{name}")
        print("-" * 80)

        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []

        # Train
        model.train()
        for epoch in range(5):
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")

            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        results[name] = losses

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'Initial Loss':<15} {'Final Loss':<15} {'Improvement':<15}")
    print("-" * 80)

    for name, losses in results.items():
        initial = losses[0]
        final = losses[-1]
        improvement = ((initial - final) / initial) * 100

        print(f"{name:<30} {initial:<15.4f} {final:<15.4f} {improvement:<15.1f}%")

    print("=" * 80)
    print()

    # Analysis
    with_pos_final = results["With Position (Sinusoidal)"][-1]
    without_pos_final = results["WITHOUT Position"][-1]

    difference = ((without_pos_final - with_pos_final) / with_pos_final) * 100

    print("CONCLUSION:")
    print(f"  Model WITHOUT position performs {difference:.1f}% WORSE")
    print(f"  Position information is CRITICAL for sequence modeling!")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: No Position")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with vs without position (slower)")
    args = parser.parse_args()

    demonstrate_no_position()

    if args.compare:
        compare_with_vs_without_position()


if __name__ == "__main__":
    main()
