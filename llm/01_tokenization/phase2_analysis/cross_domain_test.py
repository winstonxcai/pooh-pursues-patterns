"""
Cross-Domain Tokenization Analysis

Tests BPE tokenization efficiency across different text types:
- English prose
- Python code
- Numeric strings
- URLs

Also analyzes effect of vocabulary size on tokenization efficiency.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import test data
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'data/raw/01_tokenization'))
from test_data import ENGLISH_TEXT, PYTHON_CODE, NUMERIC_TEXT, URL_TEXT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


class BPETokenizerSimple:
    """Lightweight BPE tokenizer for analysis."""

    def __init__(self, merges: List[Tuple[str, str]]):
        self.merges = merges

    def tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using BPE merges."""
        word_tokens = list(word) + ['</w>']

        for merge_pair in self.merges:
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

        return word_tokens

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize entire text."""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        all_tokens = []

        for word in words:
            all_tokens.extend(self.tokenize_word(word))

        return all_tokens


def load_tokenizer(data_dir: Path) -> BPETokenizerSimple:
    """Load BPE tokenizer from JSON files."""
    merges_path = data_dir / 'merges.json'

    with open(merges_path, 'r') as f:
        merges_list = json.load(f)
        merges = [tuple(m) for m in merges_list]

    logger.info(f"Loaded tokenizer with {len(merges)} merges")
    return BPETokenizerSimple(merges)


def analyze_text_domain(tokenizer: BPETokenizerSimple, text: str, domain_name: str) -> Dict:
    """
    Analyze tokenization efficiency for a specific text domain.

    Args:
        tokenizer: BPE tokenizer
        text: Sample text from domain
        domain_name: Name of the domain

    Returns:
        Dictionary with analysis results
    """
    tokens = tokenizer.tokenize_text(text)
    char_count = len(text.replace(' ', '').replace('\n', ''))
    token_count = len(tokens)

    tokens_per_100_chars = (token_count / char_count) * 100 if char_count > 0 else 0

    logger.info(f"{domain_name}: {token_count} tokens for {char_count} chars "
                f"({tokens_per_100_chars:.2f} tokens/100 chars)")

    return {
        'domain': domain_name,
        'char_count': char_count,
        'token_count': token_count,
        'tokens_per_100_chars': tokens_per_100_chars,
        'compression_ratio': char_count / token_count if token_count > 0 else 0
    }


def visualize_cross_domain_comparison(results: List[Dict], output_path: Path):
    """
    Create bar chart comparing tokenization efficiency across domains.

    Args:
        results: List of analysis results per domain
        output_path: Where to save the plot
    """
    logger.info("Creating cross-domain comparison visualization")

    domains = [r['domain'] for r in results]
    tokens_per_100 = [r['tokens_per_100_chars'] for r in results]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(domains, tokens_per_100, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                   edgecolor='black', alpha=0.8)

    plt.ylabel('Tokens per 100 Characters', fontsize=12)
    plt.xlabel('Text Domain', fontsize=12)
    plt.title('BPE Tokenization Efficiency Across Text Domains', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for bar, value in zip(bars, tokens_per_100):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-domain comparison to {output_path}")
    plt.close()


def analyze_vocabulary_size_effect(
    corpus_path: Path,
    output_path: Path,
    vocab_sizes: List[int] = [500, 1000, 2000, 5000]
):
    """
    Train BPE with different vocabulary sizes and measure efficiency.

    Args:
        corpus_path: Path to training corpus
        output_path: Where to save the plot
        vocab_sizes: List of merge counts to test
    """
    logger.info("Analyzing vocabulary size effect on tokenization efficiency")

    # Read corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use last 1000 words as test set
    all_words = re.findall(r'\w+', text.lower())
    test_words = all_words[-1000:] if len(all_words) > 1000 else all_words
    test_text = ' '.join(test_words)

    results = []

    for num_merges in vocab_sizes:
        logger.info(f"Testing with {num_merges} merges")

        # Load or train tokenizer with this many merges
        tokenizer_dir = Path(f'data/processed/01_tokenization_vocab{num_merges}')

        if not (tokenizer_dir / 'merges.json').exists():
            logger.info(f"Training tokenizer with {num_merges} merges")
            # Train new tokenizer
            # Import from phase1_bpe
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / 'phase1_bpe'))
            from train_bpe import BPETokenizer
            sys.path.pop(0)

            tokenizer_full = BPETokenizer()
            tokenizer_full.train(text, num_merges=num_merges)
            tokenizer_full.save(tokenizer_dir)

        # Load and test
        tokenizer = load_tokenizer(tokenizer_dir)

        # Measure efficiency on test set
        tokens = tokenizer.tokenize_text(test_text)
        avg_tokens_per_word = len(tokens) / len(test_words) if test_words else 0

        results.append({
            'vocab_size': num_merges,
            'avg_tokens_per_word': avg_tokens_per_word
        })

        logger.info(f"  {num_merges} merges -> {avg_tokens_per_word:.3f} tokens/word")

    # Plot
    plt.figure(figsize=(10, 6))
    vocab_sizes_plot = [r['vocab_size'] for r in results]
    avg_tokens = [r['avg_tokens_per_word'] for r in results]

    plt.plot(vocab_sizes_plot, avg_tokens, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('Vocabulary Size (Number of Merges)', fontsize=12)
    plt.ylabel('Average Tokens per Word', fontsize=12)
    plt.title('Vocabulary Size vs Tokenization Efficiency', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved vocabulary size analysis to {output_path}")
    plt.close()


def create_pathological_examples_table(tokenizer: BPETokenizerSimple, output_path: Path):
    """
    Display pathological tokenization cases.

    Args:
        tokenizer: BPE tokenizer
        output_path: Where to save the visualization
    """
    logger.info("Creating pathological examples table")

    examples = [
        ("Phone number", "123-456-7890"),
        ("URL", "https://example.com/path?query=value"),
        ("Chinese", "未来世界"),
        ("Long rare word", "supercalifragilisticexpialidocious"),
        ("Mixed case code", "CamelCaseVariableName"),
        ("Email", "user@example.com")
    ]

    # Create table data
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = [["Category", "Input Text", "Tokenization", "Token Count"]]

    for category, text in examples:
        tokens = tokenizer.tokenize_text(text)
        # Create visual representation
        token_str = ' | '.join(tokens[:15])  # Limit to first 15 tokens
        if len(tokens) > 15:
            token_str += " ..."

        table_data.append([
            category,
            text,
            token_str,
            str(len(tokens))
        ])

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.15, 0.25, 0.45, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)

    plt.title('Pathological Tokenization Examples', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved pathological examples table to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Cross-domain tokenization analysis')
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
        help='Path to corpus for vocabulary size analysis'
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cross-domain comparison
    logger.info("Analyzing cross-domain tokenization")

    domains_data = [
        (ENGLISH_TEXT, "English Prose"),
        (PYTHON_CODE, "Python Code"),
        (NUMERIC_TEXT, "Numeric Strings"),
        (URL_TEXT, "URLs")
    ]

    results = []
    for text, name in domains_data:
        result = analyze_text_domain(tokenizer, text, name)
        results.append(result)

    visualize_cross_domain_comparison(
        results,
        output_dir / 'cross_domain_comparison.png'
    )

    # 2. Vocabulary size vs efficiency
    analyze_vocabulary_size_effect(
        Path(args.corpus),
        output_dir / 'vocab_size_vs_efficiency.png',
        vocab_sizes=[500, 1000, 2000, 5000]
    )

    # 3. Pathological examples
    create_pathological_examples_table(
        tokenizer,
        output_dir / 'pathological_examples.png'
    )

    logger.info("Cross-domain analysis complete")


if __name__ == '__main__':
    main()
