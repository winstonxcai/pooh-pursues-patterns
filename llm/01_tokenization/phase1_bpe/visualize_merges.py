"""
Visualizations for BPE Training Analysis

Creates three visualizations:
1. Merge progression - showing compression over iterations
2. Token length distribution - histogram of learned token sizes
3. Top subwords - most frequent learned multi-character tokens
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class SimpleBPETokenizer:
    """Lightweight BPE tokenizer for visualization purposes."""

    def __init__(self, vocab: Dict[str, int], merges: List[List[str]]):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]  # Convert to tuples

def load_tokenizer(data_dir: Path):
    """Load BPE tokenizer from JSON files."""
    vocab_path = data_dir / 'vocab.json'
    merges_path = data_dir / 'merges.json'

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    with open(merges_path, 'r') as f:
        merges = json.load(f)

    tokenizer = SimpleBPETokenizer(vocab, merges)
    logger.info(f"Loaded tokenizer with {len(tokenizer.vocab)} tokens and {len(tokenizer.merges)} merges")
    return tokenizer


def visualize_merge_progression(
    tokenizer,
    test_text: str,
    output_path: Path,
    num_samples: int = 100
):
    """
    Visualize how average tokens per word decreases with merge iterations.

    Args:
        tokenizer: Trained BPE tokenizer
        test_text: Sample text to test on
        output_path: Where to save the plot
        num_samples: Number of merge checkpoints to sample
    """
    logger.info("Creating merge progression visualization")

    # Get sample text words
    words = re.findall(r'\w+', test_text.lower())[:1000]  # Use first 1000 words

    # Sample merge checkpoints
    total_merges = len(tokenizer.merges)
    if total_merges < num_samples:
        sample_points = list(range(0, total_merges + 1))
    else:
        step = max(1, total_merges // num_samples)
        sample_points = list(range(0, total_merges + 1, step))

    avg_tokens_per_word = []

    for num_merges in sample_points:
        # Create temporary tokenizer with limited merges
        total_tokens = 0
        for word in words:
            word_tokens = list(word) + ['</w>']

            # Apply only first num_merges
            for merge_pair in tokenizer.merges[:num_merges]:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i] == merge_pair[0] and
                            word_tokens[i + 1] == merge_pair[1]):
                        word_tokens = (
                            word_tokens[:i] +
                            [merge_pair[0] + merge_pair[1]] +
                            word_tokens[i + 2:]
                        )
                    else:
                        i += 1

            total_tokens += len(word_tokens)

        avg_tokens_per_word.append(total_tokens / len(words))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(sample_points, avg_tokens_per_word, linewidth=2, color='#2E86AB')
    plt.xlabel('Number of Merges', fontsize=12)
    plt.ylabel('Average Tokens per Word', fontsize=12)
    plt.title('BPE Merge Progression: Compression Improves with More Merges', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved merge progression plot to {output_path}")
    plt.close()


def visualize_token_length_distribution(vocab: Dict[str, int], output_path: Path):
    """
    Create histogram of token lengths in the vocabulary.

    Args:
        vocab: Token to ID mapping
        output_path: Where to save the plot
    """
    logger.info("Creating token length distribution visualization")

    # Calculate token lengths (excluding </w> marker in counting)
    token_lengths = []
    for token in vocab.keys():
        # Remove </w> for length calculation
        clean_token = token.replace('</w>', '')
        if clean_token:  # Ignore empty tokens
            token_lengths.append(len(clean_token))

    # Create bins
    max_len = max(token_lengths)
    bins = list(range(1, min(max_len + 2, 15)))  # Cap at 14+ for readability

    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=bins, edgecolor='black', alpha=0.7, color='#A23B72')
    plt.xlabel('Token Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Token Lengths in Learned Vocabulary', fontsize=14, fontweight='bold')
    plt.xticks(bins)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved token length distribution to {output_path}")
    plt.close()


def visualize_top_subwords(
    tokenizer,
    corpus_text: str,
    output_path: Path,
    top_n: int = 30
):
    """
    Show most frequent learned multi-character subwords.

    Args:
        tokenizer: Trained BPE tokenizer
        corpus_text: Original corpus text
        output_path: Where to save the plot
        top_n: Number of top subwords to display
    """
    logger.info("Creating top subwords visualization")

    # Tokenize the entire corpus
    words = re.findall(r'\w+', corpus_text.lower())

    # Count subword occurrences
    subword_counts = Counter()

    for word in words:
        word_tokens = list(word) + ['</w>']

        # Apply all merges
        for merge_pair in tokenizer.merges:
            i = 0
            while i < len(word_tokens) - 1:
                if (word_tokens[i] == merge_pair[0] and
                        word_tokens[i + 1] == merge_pair[1]):
                    word_tokens = (
                        word_tokens[:i] +
                        [merge_pair[0] + merge_pair[1]] +
                        word_tokens[i + 2:]
                    )
                else:
                    i += 1

        # Count multi-character tokens (exclude single characters)
        for token in word_tokens:
            clean_token = token.replace('</w>', '')
            if len(clean_token) > 1:  # Only multi-character subwords
                subword_counts[token] += 1

    # Get top N
    top_subwords = subword_counts.most_common(top_n)
    tokens = [sw[0] for sw in top_subwords]
    counts = [sw[1] for sw in top_subwords]

    # Plot
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(tokens)), counts, color='#F18F01', edgecolor='black')
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    plt.xlabel('Frequency in Corpus', fontsize=12)
    plt.ylabel('Learned Subword', fontsize=12)
    plt.title(f'Top {top_n} Most Frequent Learned Subwords', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest at top
    plt.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved top subwords plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize BPE training results')
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default='data/processed/01_tokenization',
        help='Directory containing tokenizer files'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='data/raw/01_tokenization/frankenstein.txt',
        help='Path to corpus (for frequency analysis)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations/plots/01_tokenization',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = load_tokenizer(Path(args.tokenizer_dir))

    # Load corpus
    logger.info(f"Reading corpus from {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        corpus_text = f.read()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("Generating visualizations...")

    # 1. Merge progression
    visualize_merge_progression(
        tokenizer,
        corpus_text,
        output_dir / 'merge_progression.png'
    )

    # 2. Token length distribution
    visualize_token_length_distribution(
        tokenizer.vocab,
        output_dir / 'token_length_distribution.png'
    )

    # 3. Top subwords
    visualize_top_subwords(
        tokenizer,
        corpus_text,
        output_dir / 'top_subwords.png',
        top_n=30
    )

    logger.info("All visualizations complete")


if __name__ == '__main__':
    main()
