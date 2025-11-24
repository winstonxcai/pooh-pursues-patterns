"""
Shared Utilities for Positional Embeddings Project

Common functions for:
- Path management
- Device selection (MPS/CUDA/CPU)
- Data loading
- Visualization helpers
"""

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


# Project root and data paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "02_positional_embeddings"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "02_positional_embeddings"
VIZ_PLOTS_DIR = PROJECT_ROOT / "visualizations" / "plots" / "02_positional_embeddings"
VIZ_ANIMATIONS_DIR = PROJECT_ROOT / "visualizations" / "animations" / "02_positional_embeddings"

# Create directories if they don't exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VIZ_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """
    Get the best available device (MPS > CUDA > CPU).

    Returns:
        torch.device: Best available device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_toy_sequence() -> Tuple[list, dict]:
    """
    Load the toy sequence and create simple vocabulary.

    Returns:
        tokens: List of tokens
        vocab: Dict mapping tokens to IDs
    """
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    unique_tokens = sorted(set(tokens))
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    return tokens, vocab


def tokens_to_ids(tokens: list, vocab: dict) -> torch.Tensor:
    """Convert token list to tensor of IDs."""
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)


def setup_plotting_style():
    """Configure matplotlib and seaborn for consistent styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def save_figure(fig, filename: str, subdir: Optional[str] = None, dpi: int = 300):
    """
    Save figure to visualizations directory.

    Args:
        fig: matplotlib figure
        filename: Name of file (with extension)
        subdir: Optional subdirectory within plots/
        dpi: Resolution for saving
    """
    if subdir:
        save_dir = VIZ_PLOTS_DIR / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = VIZ_PLOTS_DIR

    save_path = save_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    Args:
        a: Tensor of shape (..., d)
        b: Tensor of shape (..., d)

    Returns:
        Cosine similarity scores
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1)


def plot_heatmap(data: np.ndarray, title: str, xlabel: str, ylabel: str,
                 cmap: str = 'viridis', figsize: Tuple[int, int] = (10, 8),
                 vmin: Optional[float] = None, vmax: Optional[float] = None):
    """
    Create a heatmap visualization.

    Args:
        data: 2D array to visualize
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap name
        figsize: Figure size
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        fig, ax: matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return fig, ax


def create_extended_vocabulary(n_words: int = 100) -> dict:
    """
    Create an extended vocabulary for testing.

    Args:
        n_words: Number of words in vocabulary

    Returns:
        vocab: Dictionary mapping words to IDs
    """
    # Common English words for testing
    common_words = [
        "the", "cat", "sat", "on", "mat", "a", "dog", "ran", "to", "house",
        "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "can", "and", "or", "but", "not", "with", "for", "at", "by", "from",
        "in", "out", "up", "down", "over", "under", "above", "below", "between",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "her", "its", "our", "their",
        "this", "that", "these", "those", "here", "there", "when", "where",
        "why", "how", "what", "which", "who", "whom", "whose", "all", "some",
        "many", "few", "more", "most", "less", "least", "every", "each", "any",
        "other", "another", "such", "very", "too", "so", "just", "only", "also"
    ]

    # Extend if needed
    while len(common_words) < n_words:
        common_words.append(f"word{len(common_words)}")

    return {word: idx for idx, word in enumerate(common_words[:n_words])}


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")
