"""
Byte-Pair Encoding (BPE) Tokenizer Training

Implements BPE algorithm from scratch:
1. Initialize vocabulary with characters
2. Iteratively merge most frequent character pairs
3. Save learned merge rules and vocabulary
"""

import argparse
import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained from scratch."""

    def __init__(self):
        self.vocab = {}  # token -> id mapping
        self.merges = []  # ordered list of merge operations (pair1, pair2) -> merged
        self.token_to_id = {}
        self.id_to_token = {}

    def get_word_frequencies(self, text: str) -> Dict[Tuple[str, ...], int]:
        """
        Convert text to word frequencies with character-level representation.

        Args:
            text: Input text corpus

        Returns:
            Dictionary mapping (char, char, ..., '</w>') tuples to frequencies
        """
        # Split on whitespace and punctuation while keeping words
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        word_freqs = Counter(words)
        logger.info(f"Found {len(word_freqs)} unique words from {len(words)} total tokens")

        # Convert to character sequences with end-of-word marker
        char_word_freqs = {}
        for word, freq in word_freqs.items():
            # Split into characters and add end marker
            char_word = tuple(list(word) + ['</w>'])
            char_word_freqs[char_word] = freq

        return char_word_freqs

    def get_pair_frequencies(
        self, word_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of adjacent pairs across all words.

        Args:
            word_freqs: Dictionary of character-tuple words to frequencies

        Returns:
            Dictionary mapping character pairs to total frequencies
        """
        pair_freqs = defaultdict(int)

        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq

        return dict(pair_freqs)

    def merge_pair(
        self,
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Merge all occurrences of a pair in the vocabulary.

        Args:
            word_freqs: Current word frequencies
            pair: Pair to merge (left, right)

        Returns:
            Updated word frequencies with pair merged
        """
        new_word_freqs = {}
        merged_token = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            # Find and merge all occurrences of the pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_freqs[tuple(new_word)] = freq

        return new_word_freqs

    def train(self, text: str, num_merges: int = 2000):
        """
        Train BPE tokenizer on input text.

        Args:
            text: Training corpus
            num_merges: Number of merge operations to perform
        """
        logger.info(f"Training BPE with {num_merges} merges")
        logger.info(f"Corpus size: {len(text)} characters")

        # Initialize with character-level vocabulary
        word_freqs = self.get_word_frequencies(text)

        # Perform iterative merging
        for merge_num in range(num_merges):
            # Get pair frequencies
            pair_freqs = self.get_pair_frequencies(word_freqs)

            if not pair_freqs:
                logger.warning(f"No more pairs to merge at iteration {merge_num}")
                break

            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            best_freq = pair_freqs[best_pair]

            # Merge the pair
            word_freqs = self.merge_pair(word_freqs, best_pair)
            self.merges.append(best_pair)

            if (merge_num + 1) % 100 == 0:
                logger.info(
                    f"Merge {merge_num + 1}/{num_merges}: "
                    f"{best_pair[0]} + {best_pair[1]} -> {best_pair[0] + best_pair[1]} "
                    f"(freq: {best_freq})"
                )

        # Build final vocabulary
        self._build_vocabulary(word_freqs)

        logger.info(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    def _build_vocabulary(self, word_freqs: Dict[Tuple[str, ...], int]):
        """Build token-to-id mapping from final word frequencies."""
        tokens = set()

        # Collect all unique tokens
        for word in word_freqs.keys():
            tokens.update(word)

        # Sort tokens for consistent ordering (single chars first, then by length)
        sorted_tokens = sorted(
            tokens,
            key=lambda x: (len(x), x)
        )

        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab = self.token_to_id

        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Split into words
        words = re.findall(r'\w+|[^\w\s]', text.lower())

        token_ids = []

        for word in words:
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

            # Convert to IDs
            for token in word_tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token])
                else:
                    logger.warning(f"Unknown token: {token}")

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        text = ''.join(tokens)
        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')
        return text.strip()

    def save(self, output_dir: Path):
        """Save tokenizer to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save merges
        merges_path = output_dir / 'merges.json'
        with open(merges_path, 'w') as f:
            json.dump(self.merges, f, indent=2)
        logger.info(f"Saved {len(self.merges)} merges to {merges_path}")

        # Save vocabulary
        vocab_path = output_dir / 'vocab.json'
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        logger.info(f"Saved vocabulary ({len(self.vocab)} tokens) to {vocab_path}")

        # Save full tokenizer object
        tokenizer_path = output_dir / 'tokenizer.pkl'
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved tokenizer object to {tokenizer_path}")

    @classmethod
    def load(cls, tokenizer_path: Path) -> 'BPETokenizer':
        """Load tokenizer from disk."""
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument(
        '--corpus',
        type=str,
        default='data/raw/01_tokenization/frankenstein.txt',
        help='Path to training corpus'
    )
    parser.add_argument(
        '--merges',
        type=int,
        default=2000,
        help='Number of merge operations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/01_tokenization',
        help='Output directory for tokenizer files'
    )

    args = parser.parse_args()

    # Read corpus
    logger.info(f"Reading corpus from {args.corpus}")
    corpus_path = Path(args.corpus)

    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=args.merges)

    # Save tokenizer
    output_dir = Path(args.output_dir)
    tokenizer.save(output_dir)

    # Test encoding/decoding
    test_text = "The monster approached with terrifying deliberation."
    logger.info(f"Testing tokenizer on: '{test_text}'")

    token_ids = tokenizer.encode(test_text)
    logger.info(f"Encoded to {len(token_ids)} tokens: {token_ids[:20]}...")

    decoded_text = tokenizer.decode(token_ids)
    logger.info(f"Decoded: '{decoded_text}'")

    logger.info("Training complete")


if __name__ == '__main__':
    main()
