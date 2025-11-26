"""
Permutation Invariance Test

Demonstrates that transformers without positional encoding are permutation-invariant:
- Same output for any permutation of input
- Word order doesn't matter
- Catastrophic for sequence modeling
"""

import argparse
import sys
from pathlib import Path
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    get_device,
    set_seed,
    load_toy_sequence,
    tokens_to_ids,
    setup_plotting_style,
    save_figure,
    print_section
)
from shared.toy_model import ToyTransformer


def test_permutation_invariance():
    """
    Test if model outputs are identical for permuted inputs.
    """
    print_section("Testing Permutation Invariance")

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}\n")

    # Load toy sequence
    tokens, vocab = load_toy_sequence()
    print(f"Original sequence: {' '.join(tokens)}")
    print(f"Tokens: {tokens}\n")

    # Create models
    methods = [
        ("With Position (Sinusoidal)", "sinusoidal"),
        ("WITHOUT Position", "none")
    ]

    # Test with several permutations
    print("Testing with permutations:")
    print("-" * 80)

    # Original
    original_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)

    # Create a few interesting permutations
    permutations = [
        tokens,  # Original
        ["cat", "the", "on", "sat", "mat", "the"],  # Shuffled
        ["mat", "the", "sat", "on", "cat", "the"],  # Reversed-ish
        ["the", "the", "cat", "sat", "on", "mat"],  # Move "the"s
    ]

    results = {method: [] for method, _ in methods}

    for method_name, pos_encoding in methods:
        print(f"\n{method_name}")
        print("-" * 80)

        # Create model
        model = ToyTransformer(
            vocab_size=len(vocab),
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # Get output for original
        with torch.no_grad():
            logits_original, _ = model(original_ids)
            output_original = logits_original[0]  # (seq_len, vocab_size)

        # Test permutations
        for i, perm in enumerate(permutations):
            perm_ids = torch.tensor(
                [vocab[t] for t in perm], dtype=torch.long, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits_perm, _ = model(perm_ids)
                output_perm = logits_perm[0]

            # Compute difference
            diff = torch.norm(output_original - output_perm).item()

            results[method_name].append(diff)

            print(f"  Permutation {i}: {' '.join(perm)}")
            print(f"    Difference from original: {diff:.6f}", end='')

            if diff < 1e-5:
                print(" → IDENTICAL ✓")
            else:
                print(" → DIFFERENT ✓")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    for method_name, _ in methods:
        diffs = results[method_name]
        max_diff = max(diffs)
        avg_diff = np.mean(diffs)

        print(f"{method_name}:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Avg difference: {avg_diff:.6f}")

        if max_diff < 1e-5:
            print(f"  → PERMUTATION INVARIANT (bad for sequence modeling!)")
        else:
            print(f"  → Position-dependent (good!)")
        print()

    return results, permutations


def test_all_permutations_small():
    """
    Test ALL permutations of a small sequence.
    """
    print_section("Testing ALL Permutations (Small Sequence)")

    set_seed(42)
    device = get_device()

    # Use smaller sequence for all permutations (3! = 6 permutations)
    small_tokens = ["cat", "sat", "mat"]
    small_vocab = {token: idx for idx, token in enumerate(sorted(set(small_tokens)))}

    print(f"Small sequence: {' '.join(small_tokens)}")
    print(f"Number of permutations: {len(list(itertools.permutations(small_tokens)))}\n")

    # Create models
    methods = [
        ("With Position", "sinusoidal"),
        ("Without Position", "none")
    ]

    results = {}

    for method_name, pos_encoding in methods:
        print(f"\n{method_name}")
        print("-" * 80)

        model = ToyTransformer(
            vocab_size=len(small_vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # Compute outputs for all permutations
        outputs = []
        perms_list = []

        for perm in itertools.permutations(small_tokens):
            perm_list = list(perm)
            perms_list.append(perm_list)

            perm_ids = torch.tensor(
                [small_vocab[t] for t in perm_list],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(perm_ids)
                outputs.append(logits[0].cpu().numpy())

        outputs = np.array(outputs)

        # Compute pairwise differences
        n_perms = len(outputs)
        diffs = np.zeros((n_perms, n_perms))

        for i in range(n_perms):
            for j in range(n_perms):
                diff = np.linalg.norm(outputs[i] - outputs[j])
                diffs[i, j] = diff

        # Check if all identical
        max_diff = diffs.max()
        print(f"Maximum pairwise difference: {max_diff:.6f}")

        if max_diff < 1e-5:
            print("→ ALL PERMUTATIONS PRODUCE IDENTICAL OUTPUT!")
            print("  (Model is permutation-invariant)")
        else:
            print("→ Different permutations produce different outputs")
            print("  (Model is position-dependent)")

        results[method_name] = {
            'permutations': perms_list,
            'outputs': outputs,
            'diffs': diffs
        }

    return results


def plot_permutation_variance(filename: str = "permutation_variance.png"):
    """
    Plot variance in outputs across permutations.
    """
    print("\nCreating permutation variance plot...")

    setup_plotting_style()

    set_seed(42)
    device = get_device()

    # Use a small sequence
    small_tokens = ["cat", "sat", "mat", "on"]
    small_vocab = {token: idx for idx, token in enumerate(sorted(set(small_tokens)))}

    # Random permutations
    np.random.seed(42)
    n_perms = 24  # 4! = 24 permutations total
    all_perms = list(itertools.permutations(small_tokens))

    methods = [
        ("Sinusoidal", "sinusoidal"),
        ("RoPE", "rope"),
        ("ALiBi", "alibi"),
        ("NO POSITION", "none")
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    for method_name, pos_encoding in methods:
        model = ToyTransformer(
            vocab_size=len(small_vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # Compute output variance across permutations
        variances = []

        for perm in all_perms:
            perm_ids = torch.tensor(
                [small_vocab[t] for t in perm],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(perm_ids)
                # Variance of output logits
                var = logits[0].var().item()
                variances.append(var)

        # Plot
        positions = np.arange(len(variances))
        ax.plot(positions, variances, marker='o', linewidth=1.5,
               markersize=4, label=method_name, alpha=0.7)

    ax.set_xlabel('Permutation Index')
    ax.set_ylabel('Output Variance')
    ax.set_title('Output Variance Across Permutations\n'
                '(Without position: all identical → same variance)')
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, filename, subdir='phase4')
    plt.close()


def plot_permutation_heatmap(filename: str = "permutation_heatmap.png"):
    """
    Plot heatmap of pairwise differences between permutations.
    """
    print("Creating permutation difference heatmap...")

    setup_plotting_style()

    set_seed(42)
    device = get_device()

    small_tokens = ["cat", "sat", "mat"]
    small_vocab = {token: idx for idx, token in enumerate(sorted(set(small_tokens)))}

    methods = [
        ("With Position (Sinusoidal)", "sinusoidal"),
        ("WITHOUT Position", "none")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (method_name, pos_encoding) in zip(axes, methods):
        model = ToyTransformer(
            vocab_size=len(small_vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # All permutations
        perms = list(itertools.permutations(small_tokens))
        outputs = []

        for perm in perms:
            perm_ids = torch.tensor(
                [small_vocab[t] for t in perm],
                dtype=torch.long,
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(perm_ids)
                outputs.append(logits[0].cpu().numpy())

        # Compute pairwise differences
        n = len(outputs)
        diffs = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                diffs[i, j] = np.linalg.norm(outputs[i] - outputs[j])

        # Plot
        im = ax.imshow(diffs, cmap='RdYlGn_r', aspect='auto')
        ax.set_xlabel('Permutation Index')
        ax.set_ylabel('Permutation Index')
        ax.set_title(f'{method_name}\n(Max diff: {diffs.max():.4f})')

        plt.colorbar(im, ax=ax, label='Difference')

        # Add permutation labels
        perm_labels = [' '.join(p[:2]) + '...' for p in perms]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(perm_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(perm_labels, fontsize=8)

    plt.suptitle('Pairwise Differences Between Permutations', fontsize=16)
    plt.tight_layout()
    save_figure(fig, filename, subdir='phase4')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test Permutation Invariance")
    parser.add_argument("--all-perms", action="store_true",
                       help="Test all permutations (slow for long sequences)")
    args = parser.parse_args()

    # Basic test
    test_permutation_invariance()

    # All permutations test
    if args.all_perms:
        test_all_permutations_small()

    # Visualizations
    plot_permutation_variance()
    plot_permutation_heatmap()

    print("\n" + "=" * 80)
    print("Permutation tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
