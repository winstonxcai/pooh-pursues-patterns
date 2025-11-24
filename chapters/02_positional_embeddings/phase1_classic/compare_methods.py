"""
Compare Sinusoidal vs Learned Positional Embeddings

Side-by-side comparison of both methods:
- Visualizations
- Properties
- Trade-offs
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared.utils import (
    setup_plotting_style,
    save_figure,
    load_toy_sequence,
    tokens_to_ids,
    DATA_PROCESSED_DIR,
    print_section
)


def compare_encodings():
    """Compare sinusoidal and learned positional encodings."""
    print_section("Comparing Sinusoidal vs Learned Embeddings")

    setup_plotting_style()

    # Load both encodings
    print("Loading encodings...")
    sin_path = DATA_PROCESSED_DIR / "sinusoidal_encodings.pt"
    learned_path = DATA_PROCESSED_DIR / "learned_embeddings.pt"

    if not sin_path.exists():
        print(f"ERROR: {sin_path} not found. Run sinusoidal.py first.")
        return

    if not learned_path.exists():
        print(f"ERROR: {learned_path} not found. Run learned.py first.")
        return

    sin_data = torch.load(sin_path)
    learned_data = torch.load(learned_path)

    sin_encodings = sin_data['encodings'].detach().numpy()
    learned_encodings = learned_data['encodings'].detach().numpy()

    print(f"  Sinusoidal shape: {sin_encodings.shape}")
    print(f"  Learned shape: {learned_encodings.shape}\n")

    # 1. Side-by-side heatmaps
    print("Creating side-by-side heatmaps...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sinusoidal
    im1 = ax1.imshow(sin_encodings[:20, :], cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Position')
    ax1.set_title('Sinusoidal Positional Encodings')
    plt.colorbar(im1, ax=ax1)

    # Learned
    im2 = ax2.imshow(learned_encodings[:20, :], cmap='RdBu_r', aspect='auto')
    ax2.set_xlabel('Embedding Dimension')
    ax2.set_ylabel('Position')
    ax2.set_title('Learned Positional Embeddings')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    save_figure(fig, "comparison_heatmaps.png", subdir='phase1')
    plt.close()

    # 2. Side-by-side 3D projections
    print("Creating side-by-side 3D projections...")
    fig = plt.figure(figsize=(16, 7))

    max_pos = 20

    # Sinusoidal
    ax1 = fig.add_subplot(121, projection='3d')
    pca1 = PCA(n_components=3)
    sin_3d = pca1.fit_transform(sin_encodings[:max_pos])
    positions = np.arange(max_pos)

    scatter1 = ax1.scatter(sin_3d[:, 0], sin_3d[:, 1], sin_3d[:, 2],
                           c=positions, cmap='viridis', s=100, alpha=0.6)
    ax1.plot(sin_3d[:, 0], sin_3d[:, 1], sin_3d[:, 2],
             'gray', alpha=0.3, linewidth=1)
    ax1.set_xlabel(f'PC1 ({pca1.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca1.explained_variance_ratio_[1]:.1%})')
    ax1.set_zlabel(f'PC3 ({pca1.explained_variance_ratio_[2]:.1%})')
    ax1.set_title('Sinusoidal Encodings')

    # Learned
    ax2 = fig.add_subplot(122, projection='3d')
    pca2 = PCA(n_components=3)
    learned_3d = pca2.fit_transform(learned_encodings[:max_pos])

    scatter2 = ax2.scatter(learned_3d[:, 0], learned_3d[:, 1], learned_3d[:, 2],
                           c=positions, cmap='viridis', s=100, alpha=0.6)
    ax2.plot(learned_3d[:, 0], learned_3d[:, 1], learned_3d[:, 2],
             'gray', alpha=0.3, linewidth=1)
    ax2.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca2.explained_variance_ratio_[2]:.1%})')
    ax2.set_title('Learned Embeddings')

    plt.tight_layout()
    save_figure(fig, "comparison_3d.png", subdir='phase1')
    plt.close()

    # 3. Similarity distribution comparison
    print("Creating similarity distribution comparison...")
    sin_sim = cosine_similarity(sin_encodings[:max_pos])
    learned_sim = cosine_similarity(learned_encodings[:max_pos])

    # Get off-diagonal elements (exclude self-similarity)
    mask = ~np.eye(max_pos, dtype=bool)
    sin_sim_values = sin_sim[mask]
    learned_sim_values = learned_sim[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sin_sim_values, bins=50, alpha=0.6, label='Sinusoidal', density=True)
    ax.hist(learned_sim_values, bins=50, alpha=0.6, label='Learned', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Pairwise Similarities')
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, "comparison_similarity_distribution.png", subdir='phase1')
    plt.close()

    # 4. Properties table
    print("\nComparison Summary:")
    print("=" * 80)
    print(f"{'Property':<30} {'Sinusoidal':<25} {'Learned':<25}")
    print("-" * 80)
    print(f"{'Parameters':<30} {0:<25} {learned_encodings.shape[0] * learned_encodings.shape[1]:<25,}")
    print(f"{'Mean similarity':<30} {sin_sim_values.mean():.4f}{'':<20} {learned_sim_values.mean():.4f}")
    print(f"{'Std similarity':<30} {sin_sim_values.std():.4f}{'':<20} {learned_sim_values.std():.4f}")
    print(f"{'Min similarity':<30} {sin_sim_values.min():.4f}{'':<20} {learned_sim_values.min():.4f}")
    print(f"{'Max similarity':<30} {sin_sim_values.max():.4f}{'':<20} {learned_sim_values.max():.4f}")
    print(f"{'Extrapolation':<30} {'Yes (any length)':<25} {'No (limited to max_len)':<25}")
    print(f"{'Training required':<30} {'No':<25} {'Yes':<25}")
    print("=" * 80)

    # 5. Position difference analysis
    print("\nPosition Distance vs Similarity:")
    print("=" * 80)

    for distance in [1, 2, 5, 10]:
        # Collect similarities at this distance
        sin_sims_at_dist = []
        learned_sims_at_dist = []

        for i in range(max_pos - distance):
            sin_sims_at_dist.append(sin_sim[i, i + distance])
            learned_sims_at_dist.append(learned_sim[i, i + distance])

        sin_mean = np.mean(sin_sims_at_dist)
        learned_mean = np.mean(learned_sims_at_dist)

        print(f"Distance {distance:2d}: Sinusoidal={sin_mean:.4f}, Learned={learned_mean:.4f}")

    print("=" * 80)

    # 6. Combined embeddings visualization (toy sequence)
    print("\nVisualizing combined token + position embeddings...")
    tokens, vocab = load_toy_sequence()
    print(f"Toy sequence: {' '.join(tokens)}\n")

    # Create dummy token embeddings
    d_model = sin_encodings.shape[1]
    vocab_size = len(vocab)
    token_emb_matrix = np.random.randn(vocab_size, d_model) * 0.5

    # Get token IDs
    token_ids = [vocab[t] for t in tokens]

    # Combine with positional encodings
    sin_combined = np.array([token_emb_matrix[tid] + sin_encodings[pos]
                            for pos, tid in enumerate(token_ids)])
    learned_combined = np.array([token_emb_matrix[tid] + learned_encodings[pos]
                                for pos, tid in enumerate(token_ids)])

    # Project to 3D
    pca_combined = PCA(n_components=3)
    all_combined = np.vstack([sin_combined, learned_combined])
    combined_3d = pca_combined.fit_transform(all_combined)

    sin_combined_3d = combined_3d[:len(tokens)]
    learned_combined_3d = combined_3d[len(tokens):]

    fig = plt.figure(figsize=(16, 7))

    # Sinusoidal
    ax1 = fig.add_subplot(121, projection='3d')
    for i, token in enumerate(tokens):
        ax1.scatter(sin_combined_3d[i, 0], sin_combined_3d[i, 1], sin_combined_3d[i, 2],
                   s=200, alpha=0.6, label=f'{token} (pos {i})')
        ax1.text(sin_combined_3d[i, 0], sin_combined_3d[i, 1], sin_combined_3d[i, 2],
                f'{token}_{i}', fontsize=10)

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('Token + Sinusoidal Position')

    # Learned
    ax2 = fig.add_subplot(122, projection='3d')
    for i, token in enumerate(tokens):
        ax2.scatter(learned_combined_3d[i, 0], learned_combined_3d[i, 1], learned_combined_3d[i, 2],
                   s=200, alpha=0.6)
        ax2.text(learned_combined_3d[i, 0], learned_combined_3d[i, 1], learned_combined_3d[i, 2],
                f'{token}_{i}', fontsize=10)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.set_title('Token + Learned Position')

    plt.tight_layout()
    save_figure(fig, "comparison_combined_embeddings.png", subdir='phase1')
    plt.close()

    print("All comparisons complete!")


def main():
    compare_encodings()


if __name__ == "__main__":
    main()
