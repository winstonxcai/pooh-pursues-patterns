"""
Token Visualizer for BPE Analysis

Displays how text is tokenized with visual boundaries and statistics.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BPEVisualizer:
    """Visualizes BPE tokenization with boundaries and statistics."""

    def __init__(self, vocab: Dict[str, int], merges: List[List[str]]):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.token_to_id = vocab
        logger.info(f"Loaded visualizer with {len(self.vocab)} tokens and {len(self.merges)} merges")

    def tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using BPE merges.

        Args:
            word: Input word (lowercase)

        Returns:
            List of subword tokens
        """
        # Start with character-level representation
        word_tokens = list(word) + ['</w>']

        # Apply merges in order
        for merge_pair in self.merges:
            i = 0
            while i < len(word_tokens) - 1:
                if (word_tokens[i] == merge_pair[0] and
                        word_tokens[i + 1] == merge_pair[1]):
                    # Merge
                    word_tokens = (
                        word_tokens[:i] +
                        [merge_pair[0] + merge_pair[1]] +
                        word_tokens[i + 2:]
                    )
                else:
                    i += 1

        return word_tokens

    def visualize_sentence(self, text: str, show_ids: bool = True) -> Dict:
        """
        Visualize how a sentence is tokenized.

        Args:
            text: Input sentence
            show_ids: Whether to show token IDs

        Returns:
            Dictionary with visualization info
        """
        # Split into words and punctuation
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        all_tokens = []
        all_token_ids = []
        word_boundaries = []

        current_position = 0

        for word in words:
            tokens = self.tokenize_word(word)
            all_tokens.extend(tokens)

            # Get token IDs
            for token in tokens:
                token_id = self.token_to_id.get(token, -1)  # -1 for unknown
                all_token_ids.append(token_id)

            # Mark word boundary
            word_boundaries.append((current_position, current_position + len(tokens)))
            current_position += len(tokens)

        return {
            'original': text,
            'tokens': all_tokens,
            'token_ids': all_token_ids,
            'word_boundaries': word_boundaries,
            'num_tokens': len(all_tokens),
            'num_chars': len(text.replace(' ', '')),
            'compression_ratio': len(text.replace(' ', '')) / len(all_tokens) if all_tokens else 0,
            'avg_token_length': sum(len(t.replace('</w>', '')) for t in all_tokens) / len(all_tokens) if all_tokens else 0
        }

    def display_visualization(self, viz_data: Dict):
        """
        Print a nicely formatted visualization.

        Args:
            viz_data: Visualization data from visualize_sentence
        """
        tokens = viz_data['tokens']
        token_ids = viz_data['token_ids']

        # Build visual representation with separators
        token_display = []
        id_display = []

        for token, tid in zip(tokens, token_ids):
            # Clean token for display
            clean_token = token.replace('</w>', ' ')
            token_display.append(clean_token)

            # Format ID (pad to match token width)
            id_str = str(tid)
            id_display.append(id_str.center(len(clean_token)))

        # Join with separators
        visual_tokens = '|'.join(token_display)
        visual_ids = '|'.join(id_display)

        print("\n" + "=" * 80)
        print(f"Original: {viz_data['original']}")
        print("-" * 80)
        print(f"Tokens:   {visual_tokens}")
        print(f"IDs:      {visual_ids}")
        print("-" * 80)
        print(f"Statistics:")
        print(f"  Token count:        {viz_data['num_tokens']}")
        print(f"  Character count:    {viz_data['num_chars']}")
        print(f"  Compression ratio:  {viz_data['compression_ratio']:.2f} chars/token")
        print(f"  Avg token length:   {viz_data['avg_token_length']:.2f} chars")
        print("=" * 80)

    @classmethod
    def load(cls, data_dir: Path) -> 'BPEVisualizer':
        """Load visualizer from tokenizer files."""
        vocab_path = data_dir / 'vocab.json'
        merges_path = data_dir / 'merges.json'

        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        with open(merges_path, 'r') as f:
            merges = json.load(f)

        return cls(vocab, merges)


def main():
    parser = argparse.ArgumentParser(description='Visualize BPE tokenization')
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default='data/processed/01_tokenization',
        help='Directory containing tokenizer files'
    )
    parser.add_argument(
        '--sentences',
        type=str,
        nargs='+',
        help='Sentences to visualize (if not provided, uses default test sentences)'
    )

    args = parser.parse_args()

    # Load visualizer
    visualizer = BPEVisualizer.load(Path(args.tokenizer_dir))

    # Default test sentences from README
    if args.sentences:
        test_sentences = args.sentences
    else:
        test_sentences = [
            "The monster approached with terrifying deliberation.",
            "Victor felt overwhelming remorse for his creation.",
            "The desolate landscape stretched endlessly before them."
        ]

    logger.info(f"Visualizing {len(test_sentences)} sentences")

    # Visualize each sentence
    for sentence in test_sentences:
        viz_data = visualizer.visualize_sentence(sentence)
        visualizer.display_visualization(viz_data)


if __name__ == '__main__':
    main()
