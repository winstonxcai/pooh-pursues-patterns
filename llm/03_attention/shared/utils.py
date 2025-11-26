"""
Shared Utilities for Attention Project

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
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "03_attention"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "03_attention"
VIZ_PLOTS_DIR = PROJECT_ROOT / "visualizations" / "plots" / "03_attention"
VIZ_ANIMATIONS_DIR = PROJECT_ROOT / "visualizations" / "animations" / "03_attention"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "03_attention"

# Create directories if they don't exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VIZ_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_toy_sequence() -> Tuple[list, dict]:
    """
    Load the canonical toy sequence: "the cat sat on the mat"

    This 6-token sequence is used throughout the chapter for visualization
    and understanding. It has:
    - Repeated word "the" (tests positional understanding)
    - Clear semantic relationships (subject-verb-object)
    - Perfect size for attention heatmaps (6x6)

    Returns:
        tokens: List of tokens in sequence
        vocab: Dict mapping tokens to integer IDs
    """
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    # Create vocabulary from unique tokens (sorted for consistency)
    unique_tokens = sorted(set(tokens))
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}

    return tokens, vocab


def create_extended_vocabulary(n_words: int = 100) -> dict:
    """
    Create an extended vocabulary for testing on longer sequences.

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


def tokens_to_ids(tokens: list, vocab: dict) -> torch.Tensor:
    """
    Convert list of tokens to tensor of vocabulary IDs.

    Args:
        tokens: List of token strings
        vocab: Vocabulary mapping tokens to IDs

    Returns:
        Tensor of token IDs (seq_len,)
    """
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
        subdir: Optional subdirectory within plots/ (e.g., "phase1")
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


def print_section(title: str, char: str = "="):
    """
    Print a formatted section header.

    Args:
        title: Section title
        char: Character to use for border (default: "=")
    """
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def print_tensor_info(name: str, tensor: torch.Tensor):
    """
    Print helpful information about a tensor.

    Args:
        name: Name/description of tensor
        tensor: The tensor to describe
    """
    print(f"{name}:")
    print(f"  Shape: {tuple(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    if tensor.numel() <= 20:
        print(f"  Values: {tensor}")
    else:
        print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")
