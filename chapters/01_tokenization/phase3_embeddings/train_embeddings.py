"""
Skip-gram Word Embeddings Training

Trains word embeddings on BPE-tokenized text using Skip-gram objective:
- Predict context words from target word
- Learn dense vector representations
- Capture semantic relationships
"""

import argparse
import json
import logging
import pickle
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Implements:
    1. Linear warmup from min_lr to max_lr over warmup_steps
    2. Cosine annealing from max_lr to min_lr over remaining steps
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int,
                 max_lr: float, min_lr: float):
        """
        Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            max_lr: Maximum learning rate (after warmup)
            min_lr: Minimum learning rate (end of training)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate for the current step."""
        self.current_step += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))

        return lr


class BPETokenizerLoader:
    """Load BPE tokenizer for encoding text - OPTIMIZED VERSION."""

    def __init__(self, vocab: dict, merges: List[List[str]]):
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = len(vocab)
        
        # OPTIMIZATION 1: Pre-build merge pair lookup for O(1) access
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}
        
        # OPTIMIZATION 2: Pre-compile regex for word splitting
        self._word_regex = re.compile(r'\w+|[^\w\s]')
        
        # OPTIMIZATION 3: Cache for unknown tokens to avoid repeated lookups
        self._unknown_tokens = set()

    @lru_cache(maxsize=100000)  # Cache up to 100K unique words
    def tokenize_word(self, word: str) -> tuple:
        """
        Tokenize a single word using BPE with aggressive optimization.
        
        Returns tuple instead of list for hashability (required for lru_cache).
        """
        # Start with character-level tokens
        word_tokens = list(word) + ['</w>']
        
        # Continue merging until no more valid merges exist
        while len(word_tokens) > 1:
            # Build all adjacent pairs with their positions
            pairs = {}
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                if pair not in pairs and pair in self.merge_ranks:
                    pairs[pair] = i
            
            # No more valid merges
            if not pairs:
                break
            
            # Find the pair with lowest rank (highest priority)
            best_pair = min(pairs.keys(), key=lambda p: self.merge_ranks[p])
            first, second = best_pair
            
            # Merge all occurrences of this pair in one pass
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if (i < len(word_tokens) - 1 and 
                    word_tokens[i] == first and 
                    word_tokens[i + 1] == second):
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_tokens
        
        return tuple(word_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs - OPTIMIZED VERSION.
        
        Improvements:
        - Pre-compiled regex
        - Cached word tokenization
        - Batch token ID lookup
        - Optimized list building
        """
        # Use pre-compiled regex
        words = self._word_regex.findall(text.lower())
        
        # Pre-allocate approximate size (reduces reallocation)
        token_ids = []
        token_ids_extend = token_ids.extend  # Local reference (micro-optimization)
        token_to_id = self.token_to_id  # Local reference
        unknown_tokens = self._unknown_tokens  # Local reference
        
        for word in words:
            tokens = self.tokenize_word(word)
            
            # Build valid token IDs in one comprehension
            word_token_ids = []
            for token in tokens:
                if token in unknown_tokens:
                    continue
                if token in token_to_id:
                    word_token_ids.append(token_to_id[token])
                else:
                    unknown_tokens.add(token)
            
            if word_token_ids:
                token_ids_extend(word_token_ids)
        
        return token_ids
    
    def encode_fast(self, text: str) -> List[int]:
        """
        Ultra-fast encoding with aggressive optimizations.
        Use this for large datasets like WikiText-2.
        """
        words = self._word_regex.findall(text.lower())
        
        # Process in chunks for better cache locality
        chunk_size = 10000
        all_token_ids = []
        
        for chunk_start in range(0, len(words), chunk_size):
            chunk_words = words[chunk_start:chunk_start + chunk_size]
            chunk_ids = []
            
            for word in chunk_words:
                tokens = self.tokenize_word(word)
                chunk_ids.extend([
                    self.token_to_id[token] 
                    for token in tokens 
                    if token in self.token_to_id
                ])
            
            all_token_ids.extend(chunk_ids)
        
        return all_token_ids
    
    def clear_cache(self):
        """Clear the word tokenization cache."""
        self.tokenize_word.cache_clear()
        self._unknown_tokens.clear()
        logger.info("Cleared tokenization cache")

    @classmethod
    def load(cls, tokenizer_dir: Path):
        """Load tokenizer from directory."""
        with open(tokenizer_dir / 'vocab.json', 'r') as f:
            vocab = json.load(f)

        with open(tokenizer_dir / 'merges.json', 'r') as f:
            merges = json.load(f)

        logger.info(f"Loaded tokenizer with {len(vocab)} tokens and {len(merges)} merges")
        return cls(vocab, merges)


def load_wikitext2() -> str:
    """
    Load WikiText-2 dataset from HuggingFace.

    Returns:
        Combined text from train, validation, and test splits
    """
    from datasets import load_dataset

    logger.info("Loading WikiText-2 dataset from HuggingFace")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Combine all splits
    texts = []
    for split in ['train', 'validation', 'test']:
        texts.extend(dataset[split]['text'])

    # Join with newlines, filter out empty lines
    text = '\n'.join([t for t in texts if t.strip()])

    logger.info(f"Loaded WikiText-2: {len(text)} characters")
    return text


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram training with (target, context) pairs and negative sampling."""

    def __init__(self, token_ids: List[int], window_size: int = 2, 
                 num_negative_samples: int = 10, vocab_size: int = None):
        """
        Create Skip-gram training pairs with negative sampling.

        Args:
            token_ids: List of token IDs from corpus
            window_size: Context window size (tokens before and after target)
            num_negative_samples: Number of negative samples per positive pair
            vocab_size: Size of vocabulary (for negative sampling)
        """
        self.pairs = []
        self.num_negative_samples = num_negative_samples
        self.vocab_size = vocab_size or max(token_ids) + 1

        for i, target in enumerate(token_ids):
            # Get context window
            start = max(0, i - window_size)
            end = min(len(token_ids), i + window_size + 1)

            for j in range(start, end):
                if j != i:  # Skip the target itself
                    context = token_ids[j]
                    self.pairs.append((target, context))

        logger.info(f"Created {len(self.pairs)} training pairs from {len(token_ids)} tokens")
        logger.info(f"Using negative sampling with {num_negative_samples} negatives per positive")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive_context = self.pairs[idx]
        
        # Generate negative samples (random words from vocabulary)
        negative_samples = torch.randint(0, self.vocab_size, (self.num_negative_samples,))
        
        return (torch.tensor(target), 
                torch.tensor(positive_context), 
                negative_samples)


class SkipGramModel(nn.Module):
    """Skip-gram model with negative sampling for efficient training."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Skip-gram model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Target word embeddings (this is what we want to learn)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Context word embeddings (separate from target for Skip-gram)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        nn.init.uniform_(self.target_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/embedding_dim, 0.5/embedding_dim)

    def forward(self, target_ids, positive_ids, negative_ids):
        """
        Forward pass with negative sampling.

        Args:
            target_ids: Target word IDs (batch_size,)
            positive_ids: Positive context word IDs (batch_size,)
            negative_ids: Negative sample IDs (batch_size, num_negative_samples)

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        # Get target embeddings (batch_size, embed_dim)
        target_embeds = self.target_embeddings(target_ids)
        
        # Get positive context embeddings (batch_size, embed_dim)
        positive_embeds = self.context_embeddings(positive_ids)
        
        # Get negative context embeddings (batch_size, num_neg, embed_dim)
        negative_embeds = self.context_embeddings(negative_ids)
        
        # Compute positive scores: batch_size
        positive_scores = (target_embeds * positive_embeds).sum(dim=1)
        
        # Compute negative scores: (batch_size, num_neg)
        negative_scores = torch.bmm(negative_embeds, target_embeds.unsqueeze(2)).squeeze(2)
        
        return positive_scores, negative_scores

    def get_embeddings(self) -> np.ndarray:
        """Get learned target embeddings as numpy array."""
        return self.target_embeddings.weight.detach().cpu().numpy()


def train_skipgram(
    token_ids: List[int],
    vocab_size: int,
    embedding_dim: int = 100,
    window_size: int = 5,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 0.025,
    min_lr: float = 0.001,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    num_negative_samples: int = 10,
    device: str = 'cpu'
) -> Tuple[SkipGramModel, List[float]]:
    """
    Train Skip-gram embeddings with negative sampling and learning rate scheduling.

    Args:
        token_ids: Tokenized corpus
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        window_size: Context window size (default: 5)
        epochs: Number of training epochs (default: 10)
        batch_size: Batch size (default: 512, increased for negative sampling)
        learning_rate: Maximum learning rate (default: 0.025)
        min_lr: Minimum learning rate for scheduler (default: 0.001)
        warmup_ratio: Ratio of warmup steps to total steps (default: 0.1)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        num_negative_samples: Number of negative samples per positive (default: 10)
        device: Device to train on

    Returns:
        Trained model and loss history
    """
    logger.info(f"Training Skip-gram model with NEGATIVE SAMPLING")
    logger.info(f"  vocab_size={vocab_size}, embedding_dim={embedding_dim}, window_size={window_size}")
    logger.info(f"  num_negative_samples={num_negative_samples}")
    logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, "
                f"min_lr={min_lr}, warmup_ratio={warmup_ratio}")

    # Create dataset with negative sampling
    dataset = SkipGramDataset(token_ids, window_size=window_size, 
                              num_negative_samples=num_negative_samples,
                              vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Calculate total steps and warmup steps
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # Initialize model
    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    logger.info(f"Model initialized on device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=min_lr)  # Start with min_lr

    # Initialize learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=learning_rate,
        min_lr=min_lr
    )

    # Training loop
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for target_ids, positive_ids, negative_ids in progress_bar:
            target_ids = target_ids.to(device)
            positive_ids = positive_ids.to(device)
            negative_ids = negative_ids.to(device)

            # Forward pass with negative sampling
            positive_scores, negative_scores = model(target_ids, positive_ids, negative_ids)

            # Negative sampling loss
            # Positive pairs should have high scores (close to 1)
            # Negative pairs should have low scores (close to 0)
            positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-10).mean()
            negative_loss = -torch.log(1 - torch.sigmoid(negative_scores) + 1e-10).mean()
            loss = positive_loss + negative_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Update learning rate
            scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            current_lr = scheduler.get_lr()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        current_lr = scheduler.get_lr()
        logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description='Train Skip-gram embeddings')
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default='data/processed/01_tokenization',
        help='Directory containing BPE tokenizer files'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='data/raw/01_tokenization/frankenstein.txt',
        help='Training corpus (used when dataset=frankenstein)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='frankenstein',
        choices=['frankenstein', 'wikitext2'],
        help='Dataset to use for training (default: frankenstein)'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=100,
        help='Embedding dimension'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Context window size (default: 5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size (default: 512, increased for negative sampling)'
    )
    parser.add_argument(
        '--num-negative-samples',
        type=int,
        default=10,
        help='Number of negative samples per positive pair (default: 10)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.025,
        help='Maximum learning rate (default: 0.025)'
    )
    parser.add_argument(
        '--min-lr',
        type=float,
        default=0.001,
        help='Minimum learning rate for scheduler (default: 0.001)'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Ratio of warmup steps to total steps (default: 0.1)'
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (default: 1.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/checkpoints/01_tokenization',
        help='Output directory for trained model'
    )

    args = parser.parse_args()

    # Set device (prioritize MPS for Apple Silicon)
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = BPETokenizerLoader.load(Path(args.tokenizer_dir))

    # Load corpus based on dataset selection
    dataset_name = args.dataset
    if dataset_name == 'frankenstein':
        logger.info(f"Loading Frankenstein corpus from {args.corpus}")
        with open(args.corpus, 'r', encoding='utf-8') as f:
            text = f.read()
    elif dataset_name == 'wikitext2':
        text = load_wikitext2()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Corpus size: {len(text)} characters")

    token_ids = tokenizer.encode(text)
    logger.info(f"Tokenized corpus: {len(token_ids)} tokens")

    # Train model with negative sampling
    model, loss_history = train_skipgram(
        token_ids=token_ids,
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_negative_samples=args.num_negative_samples,
        device=device
    )

    # Save model and embeddings
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full model with dataset prefix
    model_path = output_dir / f'skipgram_{dataset_name}_dim{args.embedding_dim}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'embedding_dim': args.embedding_dim,
        'window_size': args.window_size,
        'loss_history': loss_history,
        'dataset': dataset_name,
        'corpus_size': len(text),
        'num_tokens': len(token_ids)
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save embeddings as numpy array with dataset prefix
    embeddings = model.get_embeddings()
    embeddings_path = output_dir / f'embeddings_{dataset_name}_dim{args.embedding_dim}.npy'
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Save token mapping for reference with dataset prefix
    token_map_path = output_dir / f'token_mapping_{dataset_name}.json'
    with open(token_map_path, 'w') as f:
        json.dump({
            'token_to_id': tokenizer.token_to_id,
            'id_to_token': tokenizer.id_to_token
        }, f, indent=2)
    logger.info(f"Saved token mapping to {token_map_path}")

    logger.info("Training complete")


if __name__ == '__main__':
    main()
