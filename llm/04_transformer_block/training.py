import sys
from pathlib import Path

# Add the current directory to the path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.transformer import TransformerBlock
from constants import (BATCH_SIZE, D_MODEL, DATA_PATH, DEVICE, DROPOUT,
                       EVAL_INTERVAL, EVAL_ITERS, LEARNING_RATE, MAX_ITERS,
                       NUM_HEADS, NUM_LAYERS, SEQ_LEN)
from processing import get_batch
from tokenizer import create_tokenizer
from tqdm import tqdm

# Load and prepare data
with open(DATA_PATH / "input.txt", "r") as f:
    text = f.read()

tokens, encode, decode = create_tokenizer(text)
vocab_size = len(tokens)

# Encode the entire text
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation sets (90% train, 10% val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

class FinalModel(nn.Module):   
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, D_MODEL)
        self.blocks = nn.Sequential(*[TransformerBlock(apply_rope=True) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights with small random values for stable training.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = idx.shape

        # Map tokens to embeddings
        x = self.embeddings(idx)

        # Apply transformer blocks
        x = self.blocks(x)

        # Apply layer normalization
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        if targets is not None:
            # Reshape logits and targets for cross-entropy loss to (batch_size * seq_len, vocab_size)
            logits = logits.view(batch_size * seq_len, logits.size(-1))
            # Reshape targets for cross-entropy loss to (batch_size * seq_len,)
            targets = targets.view(batch_size * seq_len)
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens given a context.
        
        Args:
            idx: (batch_size, seq_len) tensor of token indices
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated sequence of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to the last SEQ_LEN tokens
            idx_cond = idx[:, -SEQ_LEN:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len + 1)
        self.train()
        return idx

def estimate_loss() -> dict:
    """
    Estimate the loss on train and validation sets.
    
    Returns:
        Dictionary with 'train' and 'val' loss values
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        data_split = train_data if split == 'train' else val_data
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(data_split, split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Initialize model
model = FinalModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
pbar = tqdm(range(MAX_ITERS), desc="Training")
for steps in pbar:
    if steps % EVAL_INTERVAL == 0 or steps == MAX_ITERS - 1:
        losses = estimate_loss()
        pbar.set_postfix({
            'train_loss': f"{losses['train']:.4f}",
            'val_loss': f"{losses['val']:.4f}"
        })
        print(f"\nstep {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data, 'train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # Update progress bar with current loss
    if steps % 10 == 0:  # Update every 10 steps to avoid too frequent updates
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print("\nGenerated text:")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))