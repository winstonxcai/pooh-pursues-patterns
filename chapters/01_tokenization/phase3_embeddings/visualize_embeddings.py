"""
Embedding Visualization Suite

Creates 6 visualizations:
1. Similarity distribution (before/after training)
2. Cosine similarity heatmap
3. Nearest neighbors table
4. 2D t-SNE projection
5. Embedding dimension comparison
6. Training progress plot
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


def load_embeddings_and_vocab(model_dir: Path, embedding_dim: int = 100, dataset: str = 'frankenstein') -> Tuple[np.ndarray, Dict, Dict]:
    """
    Load trained embeddings and vocabulary mapping.

    Args:
        model_dir: Directory containing saved model
        embedding_dim: Embedding dimension to load
        dataset: Dataset name (for file naming)

    Returns:
        Embeddings array, token_to_id dict, id_to_token dict
    """
    # Load embeddings with dataset prefix
    embeddings_path = model_dir / f'embeddings_{dataset}_dim{embedding_dim}.npy'
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings: shape {embeddings.shape}")

    # Load token mapping with dataset prefix
    token_map_path = model_dir / f'token_mapping_{dataset}.json'
    with open(token_map_path, 'r') as f:
        token_map = json.load(f)

    token_to_id = token_map['token_to_id']
    # Convert string keys back to int for id_to_token
    id_to_token = {int(k): v for k, v in token_map['id_to_token'].items()}

    return embeddings, token_to_id, id_to_token


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def visualize_similarity_distribution(
    embeddings: np.ndarray,
    output_path: Path,
    sample_size: int = 1000,
    dataset_name: str = 'Frankenstein'
):
    """
    Compare similarity distributions before and after training.

    Args:
        embeddings: Trained embeddings
        output_path: Where to save the plot
        sample_size: Number of token pairs to sample
    """
    logger.info("Creating similarity distribution visualization")

    vocab_size, embed_dim = embeddings.shape

    # Sample random pairs
    np.random.seed(42)
    indices = np.random.choice(vocab_size, size=min(sample_size, vocab_size), replace=False)

    # Compute pairwise similarities for trained embeddings
    trained_sims = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = cosine_similarity(embeddings[indices[i]], embeddings[indices[j]])
            trained_sims.append(sim)

    # Generate random one-hot style embeddings (baseline)
    # One-hot embeddings have 0 similarity (orthogonal)
    random_embeddings = np.random.randn(vocab_size, embed_dim)
    random_embeddings = random_embeddings / np.linalg.norm(random_embeddings, axis=1, keepdims=True)

    random_sims = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            sim = cosine_similarity(random_embeddings[indices[i]], random_embeddings[indices[j]])
            random_sims.append(sim)

    # Plot
    plt.figure(figsize=(12, 6))

    plt.hist(random_sims, bins=50, alpha=0.5, label='Random Initialization', color='#A23B72', density=True)
    plt.hist(trained_sims, bins=50, alpha=0.5, label='After Training', color='#2E86AB', density=True)

    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Distribution of Pairwise Cosine Similarities ({dataset_name})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved similarity distribution to {output_path}")
    plt.close()


def visualize_similarity_heatmap(
    embeddings: np.ndarray,
    token_to_id: Dict,
    id_to_token: Dict,
    output_path: Path,
    selected_words: List[str] = None,
    dataset_name: str = 'Frankenstein'
):
    """
    Create heatmap of pairwise similarities for selected words.

    Args:
        embeddings: Trained embeddings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        output_path: Where to save the plot
        selected_words: List of words to include
    """
    logger.info("Creating cosine similarity heatmap")

    # Default word selection from Frankenstein
    if selected_words is None:
        selected_words = [
            'creature</w>', 'monster</w>', 'being</w>', 'wretch</w>',
            'victor</w>', 'frankenstein</w>', 'elizabeth</w>', 'clerval</w>',
            'miserable</w>', 'wretched</w>', 'desolate</w>', 'horror</w>',
            'felt</w>', 'thought</w>', 'looked</w>', 'saw</w>',
            'mountains</w>', 'night</w>', 'death</w>', 'life</w>',
            'father</w>', 'mother</w>', 'friend</w>', 'love</w>',
            'nature</w>', 'science</w>', 'power</w>', 'mind</w>',
            'heart</w>', 'soul</w>'
        ]

    # Filter to words that exist in vocabulary
    valid_words = [w for w in selected_words if w in token_to_id]
    logger.info(f"Using {len(valid_words)} words for heatmap")

    if len(valid_words) < 5:
        logger.warning("Too few valid words for heatmap")
        return

    # Get embeddings for selected words
    word_ids = [token_to_id[w] for w in valid_words]
    word_embeddings = embeddings[word_ids]

    # Compute similarity matrix
    n = len(valid_words)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_similarity(word_embeddings[i], word_embeddings[j])

    # Clean labels (remove </w>)
    labels = [w.replace('</w>', '') for w in valid_words]

    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        sim_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='coolwarm',
        center=0,
        vmin=-0.5,
        vmax=1.0,
        square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )

    plt.title(f'Pairwise Cosine Similarity Heatmap ({dataset_name})', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved similarity heatmap to {output_path}")
    plt.close()


def visualize_nearest_neighbors(
    embeddings: np.ndarray,
    token_to_id: Dict,
    id_to_token: Dict,
    output_path: Path,
    query_words: List[str] = None,
    top_k: int = 5,
    dataset_name: str = 'Frankenstein'
):
    """
    Show nearest neighbors for query words.

    Args:
        embeddings: Trained embeddings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        output_path: Where to save the visualization
        query_words: Words to find neighbors for
        top_k: Number of neighbors to show
    """
    logger.info("Creating nearest neighbors table")

    if query_words is None:
        query_words = [
            'creature</w>', 'miserable</w>', 'felt</w>',
            'victor</w>', 'death</w>', 'nature</w>',
            'friend</w>', 'power</w>'
        ]

    # Filter to existing words
    valid_queries = [w for w in query_words if w in token_to_id]

    # Build table data
    table_data = [['Query Word', 'Nearest Neighbors', 'Similarities']]

    for query in valid_queries:
        query_id = token_to_id[query]
        query_embed = embeddings[query_id]

        # Compute similarities to all other words
        similarities = []
        for token_id in range(len(embeddings)):
            if token_id != query_id:
                sim = cosine_similarity(query_embed, embeddings[token_id])
                similarities.append((token_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k
        top_neighbors = similarities[:top_k]
        neighbor_tokens = [id_to_token[tid].replace('</w>', '') for tid, _ in top_neighbors]
        neighbor_sims = [f"{sim:.3f}" for _, sim in top_neighbors]

        table_data.append([
            query.replace('</w>', ''),
            ', '.join(neighbor_tokens),
            ', '.join(neighbor_sims)
        ])

    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, len(valid_queries) + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.15, 0.55, 0.30]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(3):
            table[(i, j)].set_facecolor(color)

    plt.title(f'Nearest Neighbors in Embedding Space ({dataset_name})', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved nearest neighbors table to {output_path}")
    plt.close()


def visualize_tsne_projection(
    embeddings: np.ndarray,
    token_to_id: Dict,
    id_to_token: Dict,
    output_path: Path,
    num_words: int = 200,
    perplexity: int = 30,
    dataset_name: str = 'Frankenstein'
):
    """
    Create 2D t-SNE projection of embeddings.

    Args:
        embeddings: Trained embeddings
        token_to_id: Token to ID mapping
        id_to_token: ID to token mapping
        output_path: Where to save the plot
        num_words: Number of words to visualize
        perplexity: t-SNE perplexity parameter
    """
    logger.info(f"Creating t-SNE projection (this may take a minute)")

    # Select most common tokens (lower IDs tend to be more common)
    # Skip very low IDs which are single characters
    start_idx = 50  # Skip single chars
    selected_ids = list(range(start_idx, min(start_idx + num_words, len(embeddings))))

    selected_embeddings = embeddings[selected_ids]
    selected_tokens = [id_to_token[i].replace('</w>', '') for i in selected_ids]

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(selected_embeddings)

    # Plot
    plt.figure(figsize=(16, 12))

    # Scatter points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=30, c='#2E86AB')

    # Add labels for a subset of points
    label_every = max(1, len(selected_tokens) // 50)  # Label ~50 points
    for i in range(0, len(selected_tokens), label_every):
        plt.annotate(
            selected_tokens[i],
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f'2D t-SNE Projection of Token Embeddings ({dataset_name})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved t-SNE projection to {output_path}")
    plt.close()


def visualize_dimension_comparison(
    model_dir: Path,
    output_path: Path,
    dimensions: List[int] = [10, 25, 50, 100, 200],
    dataset: str = 'frankenstein',
    dataset_name: str = 'Frankenstein'
):
    """
    Compare embedding quality across dimensions.

    Args:
        model_dir: Directory containing models of different dimensions
        output_path: Where to save the plot
        dimensions: List of embedding dimensions to compare
    """
    logger.info("Creating embedding dimension comparison")

    # Load embeddings and compute average similarity for each dimension
    results = []

    for dim in dimensions:
        embeddings_path = model_dir / f'embeddings_{dataset}_dim{dim}.npy'

        if not embeddings_path.exists():
            logger.warning(f"Embeddings for dim={dim} not found, skipping")
            continue

        embeddings = np.load(embeddings_path)

        # Sample some pairs and compute average similarity
        np.random.seed(42)
        sample_size = 500
        vocab_size = min(500, len(embeddings))
        indices = np.random.choice(len(embeddings), size=vocab_size, replace=False)

        sims = []
        for i in range(len(indices)):
            for j in range(i + 1, min(i + 20, len(indices))):  # Limit pairs
                sim = cosine_similarity(embeddings[indices[i]], embeddings[indices[j]])
                sims.append(sim)

        avg_sim = np.mean(sims)
        std_sim = np.std(sims)

        results.append({
            'dimension': dim,
            'avg_similarity': avg_sim,
            'std_similarity': std_sim
        })

        logger.info(f"Dim {dim}: avg_sim={avg_sim:.3f}, std_sim={std_sim:.3f}")

    if not results:
        logger.warning("No embedding files found for dimension comparison")
        return

    # Plot
    dims = [r['dimension'] for r in results]
    avg_sims = [r['avg_similarity'] for r in results]
    std_sims = [r['std_similarity'] for r in results]

    plt.figure(figsize=(10, 6))

    plt.errorbar(dims, avg_sims, yerr=std_sims, marker='o', linewidth=2,
                 markersize=8, capsize=5, color='#2E86AB', label='Avg. Pairwise Similarity')

    plt.xlabel('Embedding Dimension', fontsize=12)
    plt.ylabel('Average Cosine Similarity', fontsize=12)
    plt.title(f'Embedding Dimension vs. Learned Similarity ({dataset_name})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved dimension comparison to {output_path}")
    plt.close()


def visualize_training_progress(
    model_dir: Path,
    output_path: Path,
    embedding_dim: int = 100,
    dataset: str = 'frankenstein',
    dataset_name: str = 'Frankenstein'
):
    """
    Plot training loss over epochs.

    Args:
        model_dir: Directory containing saved model
        output_path: Where to save the plot
        embedding_dim: Embedding dimension
    """
    logger.info("Creating training progress visualization")

    # Load model checkpoint with loss history
    model_path = model_dir / f'skipgram_{dataset}_dim{embedding_dim}.pt'

    if not model_path.exists():
        logger.warning(f"Model checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location='cpu')
    loss_history = checkpoint.get('loss_history', [])

    if not loss_history:
        logger.warning("No loss history found in checkpoint")
        return

    # Plot
    plt.figure(figsize=(10, 6))

    epochs = list(range(1, len(loss_history) + 1))
    plt.plot(epochs, loss_history, marker='o', linewidth=2, markersize=8, color='#2E86AB')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(f'Training Progress: Loss Over Epochs ({dataset_name})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (e, loss) in enumerate(zip(epochs, loss_history)):
        plt.text(e, loss, f'{loss:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved training progress to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize trained embeddings')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/checkpoints/01_tokenization',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=100,
        help='Embedding dimension to visualize'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='frankenstein',
        choices=['frankenstein', 'wikitext2'],
        help='Dataset to visualize (default: frankenstein)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/plots/01_tokenization',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset info
    dataset = args.dataset
    dataset_name = 'Frankenstein' if dataset == 'frankenstein' else 'WikiText-2'

    logger.info(f"Visualizing {dataset_name} embeddings")

    # Load embeddings
    embeddings, token_to_id, id_to_token = load_embeddings_and_vocab(
        model_dir, args.embedding_dim, dataset
    )

    logger.info("Generating visualizations...")

    # 1. Similarity distribution
    visualize_similarity_distribution(
        embeddings,
        output_dir / f'{dataset}_embedding_similarity_distribution.png',
        dataset_name=dataset_name
    )

    # 2. Similarity heatmap
    visualize_similarity_heatmap(
        embeddings,
        token_to_id,
        id_to_token,
        output_dir / f'{dataset}_embedding_similarity_heatmap.png',
        dataset_name=dataset_name
    )

    # 3. Nearest neighbors
    visualize_nearest_neighbors(
        embeddings,
        token_to_id,
        id_to_token,
        output_dir / f'{dataset}_embedding_nearest_neighbors.png',
        dataset_name=dataset_name
    )

    # 4. t-SNE projection
    visualize_tsne_projection(
        embeddings,
        token_to_id,
        id_to_token,
        output_dir / f'{dataset}_embedding_tsne_projection.png',
        dataset_name=dataset_name
    )

    # 5. Dimension comparison
    visualize_dimension_comparison(
        model_dir,
        output_dir / f'{dataset}_embedding_dimension_comparison.png',
        dataset=dataset,
        dataset_name=dataset_name
    )

    # 6. Training progress
    visualize_training_progress(
        model_dir,
        output_dir / f'{dataset}_embedding_training_progress.png',
        args.embedding_dim,
        dataset=dataset,
        dataset_name=dataset_name
    )

    logger.info(f"All {dataset_name} embedding visualizations complete")


if __name__ == '__main__':
    main()
