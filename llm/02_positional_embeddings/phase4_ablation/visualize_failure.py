"""
Visualize Attention Failure Without Position

Shows how attention degrades without positional information:
- Attention heatmaps (with vs without position)
- Embedding space collapse
- Performance degradation
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.toy_model import ToyTransformer
from shared.utils import (get_device, load_toy_sequence, print_section,
                          save_figure, set_seed, setup_plotting_style,
                          tokens_to_ids)


def plot_attention_comparison(seq_len: int = 12, filename: str = "no_position_attention_comparison.png"):
    """
    Compare attention patterns with and without positional encoding.

    Args:
        seq_len: Sequence length
        filename: Output filename
    """
    print("Creating attention comparison...")

    set_seed(42)
    device = get_device()

    vocab_size = 20
    d_model = 64

    # Create dummy input
    inputs = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Methods to compare
    methods = [
        ("Sinusoidal", "sinusoidal"),
        ("RoPE", "rope"),
        ("ALiBi", "alibi"),
        ("NO POSITION", "none")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, pos_encoding) in enumerate(methods):
        ax = axes[idx]

        # Create model
        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # Get attention weights
        with torch.no_grad():
            _, attention_weights = model(inputs, return_attention=True)

        # Average over heads
        attn = attention_weights[0][0].mean(dim=0).cpu().numpy()

        # Plot
        im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'{name}')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add grid
        ax.grid(alpha=0.2, color='white')

    plt.suptitle('Attention Patterns: With vs Without Positional Information',
                fontsize=16, y=1.02)
    plt.tight_layout()
    save_figure(fig, filename, subdir='phase4')
    plt.close()


def plot_embedding_collapse(filename: str = "no_position_embedding_collapse.png"):
    """
    Show how embeddings collapse without position information.

    Demonstrates that identical tokens have identical representations.
    """
    print("Creating embedding collapse visualization...")

    set_seed(42)
    device = get_device()

    # Create sentence with repeated words
    tokens, vocab = load_toy_sequence()  # "the cat sat on the mat"
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)

    d_model = 64
    methods = [
        ("With Position (Sinusoidal)", "sinusoidal"),
        ("WITHOUT Position", "none")
    ]

    fig = plt.figure(figsize=(14, 6))

    for idx, (name, pos_encoding) in enumerate(methods):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

        # Create model
        model = ToyTransformer(
            vocab_size=len(vocab),
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        # Get embeddings
        with torch.no_grad():
            embeddings = model.get_embeddings(token_ids)[0].cpu().numpy()

        # PCA to 3D
        pca = PCA(n_components=3)
        emb_3d = pca.fit_transform(embeddings)

        # Plot
        colors = plt.cm.tab10(np.arange(len(tokens)))
        for i, (token, pos) in enumerate(zip(tokens, emb_3d)):
            ax.scatter(pos[0], pos[1], pos[2], c=[colors[i]], s=200, alpha=0.6)
            ax.text(pos[0], pos[1], pos[2], f'{token}_{i}', fontsize=10)

        # Highlight repeated "the" and compute distances
        the_indices = [i for i, t in enumerate(tokens) if t == "the"]
        if len(the_indices) > 1:
            the_positions = emb_3d[the_indices]
            ax.plot(the_positions[:, 0], the_positions[:, 1], the_positions[:, 2],
                   'r--', linewidth=2, alpha=0.5, label='Same word "the"')
            
            # Compute actual distance in original 64D space
            the_embeddings_64d = embeddings[the_indices]
            dist_64d = np.linalg.norm(the_embeddings_64d[0] - the_embeddings_64d[1])
            
            # Compute distance in 3D projection
            dist_3d = np.linalg.norm(the_positions[0] - the_positions[1])
            
            print(f"\n{name}:")
            print(f"  Distance between 'the_0' and 'the_4' in 64D: {dist_64d:.4f}")
            print(f"  Distance between 'the_0' and 'the_4' in 3D projection: {dist_3d:.4f}")

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        # Add distance info to title if we have repeated words
        if len(the_indices) > 1:
            the_embeddings_64d = embeddings[the_indices]
            dist_64d = np.linalg.norm(the_embeddings_64d[0] - the_embeddings_64d[1])
            ax.set_title(f'{name}\n("the" distance: {dist_64d:.2f})')
        else:
            ax.set_title(name)
        
        ax.legend()

    plt.suptitle('Embedding Space: Position Separates Identical Tokens', fontsize=16)
    plt.tight_layout()
    save_figure(fig, filename, subdir='phase4')
    plt.close()


def plot_attention_entropy(seq_len: int = 12, filename: str = "no_position_attention_entropy.png"):
    """
    Plot attention entropy with and without position.

    Lower entropy = more focused attention.
    """
    print("Creating attention entropy comparison...")

    set_seed(42)
    device = get_device()

    vocab_size = 20
    d_model = 64

    # Create dummy input
    inputs = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Methods to compare
    methods = ["sinusoidal", "rope", "alibi", "none"]
    method_names = ["Sinusoidal", "RoPE", "ALiBi", "NO POSITION"]

    entropies_per_method = []

    for pos_encoding in methods:
        model = ToyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_encoding
        ).to(device)

        model.eval()

        with torch.no_grad():
            _, attention_weights = model(inputs, return_attention=True)

        # Compute entropy for each query position
        attn = attention_weights[0][0].mean(dim=0).cpu().numpy()  # Average over heads

        # Entropy: -sum(p * log(p))
        entropies = []
        for i in range(seq_len):
            p = attn[i]
            # Avoid log(0)
            p = np.clip(p, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p))
            entropies.append(entropy)

        entropies_per_method.append(entropies)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    positions = np.arange(seq_len)
    colors = ['blue', 'green', 'purple', 'red']

    for entropies, name, color in zip(entropies_per_method, method_names, colors):
        ax.plot(positions, entropies, marker='o', linewidth=2, markersize=8,
               label=name, color=color)

    ax.set_xlabel('Query Position')
    ax.set_ylabel('Attention Entropy (nats)')
    ax.set_title('Attention Entropy: With vs Without Position\n(Lower = More Focused)')
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, filename, subdir='phase4')
    plt.close()


def plot_position_effect_summary(filename: str = "no_position_summary.png"):
    """
    Create summary visualization showing all effects of removing position.
    """
    print("Creating summary visualization...")

    set_seed(42)
    device = get_device()

    tokens, vocab = load_toy_sequence()
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)
    seq_len = len(tokens)

    fig = plt.figure(figsize=(18, 10))

    # 1. Attention heatmap comparison
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)

    for ax, pos_enc, title in [(ax1, "sinusoidal", "With Position"),
                               (ax2, "none", "Without Position")]:
        model = ToyTransformer(
            vocab_size=len(vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_enc
        ).to(device)

        model.eval()

        with torch.no_grad():
            _, attn_weights = model(token_ids, return_attention=True)

        attn = attn_weights[0][0].mean(dim=0).cpu().numpy()

        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f'Attention: {title}')
        plt.colorbar(im, ax=ax)

    # 3. Attention distribution for middle query
    ax3 = plt.subplot(2, 3, 3)
    query_pos = seq_len // 2

    for pos_enc, label, color in [("sinusoidal", "With Position", "blue"),
                                  ("none", "Without Position", "red")]:
        model = ToyTransformer(
            vocab_size=len(vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_enc
        ).to(device)

        model.eval()

        with torch.no_grad():
            _, attn_weights = model(token_ids, return_attention=True)

        attn = attn_weights[0][0].mean(dim=0).cpu().numpy()

        ax3.plot(np.arange(seq_len), attn[query_pos], marker='o',
                linewidth=2, label=label, color=color)

    ax3.axvline(x=query_pos, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Key Position')
    ax3.set_ylabel('Attention Weight')
    ax3.set_title(f'Attention from Query Position {query_pos}')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4-6. Embedding visualization
    for idx, (pos_enc, title) in enumerate([("sinusoidal", "With Position"),
                                             ("none", "Without Position")]):
        ax = plt.subplot(2, 3, 4 + idx, projection='3d')

        model = ToyTransformer(
            vocab_size=len(vocab),
            d_model=64,
            n_heads=4,
            n_layers=1,
            max_seq_len=128,
            position_encoding=pos_enc
        ).to(device)

        model.eval()

        with torch.no_grad():
            embeddings = model.get_embeddings(token_ids)[0].cpu().numpy()

        pca = PCA(n_components=3)
        emb_3d = pca.fit_transform(embeddings)

        colors_scatter = plt.cm.tab10(np.arange(len(tokens)))
        for i, (token, pos) in enumerate(zip(tokens, emb_3d)):
            ax.scatter(pos[0], pos[1], pos[2], c=[colors_scatter[i]], s=150, alpha=0.6)
            ax.text(pos[0], pos[1], pos[2], f'{token}_{i}', fontsize=8)

        # Compute distance for "the" tokens
        the_indices = [i for i, t in enumerate(tokens) if t == "the"]
        if len(the_indices) > 1:
            the_embeddings_64d = embeddings[the_indices]
            dist_64d = np.linalg.norm(the_embeddings_64d[0] - the_embeddings_64d[1])
            ax.set_title(f'Embeddings: {title}\n("the" distance: {dist_64d:.2f})')
        else:
            ax.set_title(f'Embeddings: {title}')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

    plt.suptitle('Effects of Removing Positional Information', fontsize=18, y=0.98)
    plt.tight_layout()
    save_figure(fig, filename, subdir='phase4')
    plt.close()


def visualize_all():
    """Generate all Phase 4 visualizations."""
    print_section("Generating Phase 4 (Ablation) Visualizations")

    setup_plotting_style()

    plot_attention_comparison()
    plot_embedding_collapse()
    plot_attention_entropy()
    plot_position_effect_summary()

    print("\n" + "=" * 80)
    print("All ablation visualizations complete!")
    print(f"Check visualizations/plots/phase4/ for plots")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Visualize No Position Failure")
    args = parser.parse_args()

    visualize_all()


if __name__ == "__main__":
    main()
