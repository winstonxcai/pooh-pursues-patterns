"""
Phase 1: Train Single-Head Self-Attention on Next-Token Prediction

Trains the self-attention model to predict the next token in a sequence.
This allows us to observe how attention patterns evolve from random (untrained)
to focused and meaningful (trained).

Training Task: Given tokens [t0, t1, ..., t_{n-1}], predict t_n
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from self_attention import SelfAttention
from shared.utils import (DATA_PROCESSED_DIR, get_device, print_section,
                          print_tensor_info, set_seed)


class SimpleLanguageModel(nn.Module):
    """
    Simple language model with single-head self-attention for next-token prediction.
    
    Architecture:
        Token Embedding → Self-Attention → LayerNorm → FFN → Output Projection
    
    Uses the same SelfAttention module from self_attention.py.
    """
    
    def __init__(self, vocab_size: int, d_model: int, d_k: int, d_ff: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Self-attention module (imported from self_attention.py)
        self.self_attn = SelfAttention(d_model, d_k, dropout=0.0)
        
        # Layer norm
        self.ln1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, token_ids, mask=None, return_attention=False):
        """
        Forward pass with self-attention.
        
        Args:
            token_ids: (batch, seq_len) - Token IDs
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch, seq_len, vocab_size) - Output logits
            attention_weights: (batch, seq_len, seq_len) - Optional attention weights
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings: (batch, seq_len, d_model)
        x = self.embedding(token_ids)
        
        # Self-attention using the imported module
        if return_attention:
            attn_output, attention_weights = self.self_attn(x, mask=mask, return_attention=True)
        else:
            attn_output = self.self_attn(x, mask=mask, return_attention=False)
            attention_weights = None
        
        # Residual connection + layer norm
        x = self.ln1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)
        
        if return_attention:
            return logits, attention_weights
        return logits


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal mask for autoregressive training.
    
    Returns:
        mask: (seq_len, seq_len) with 1=allowed, 0=blocked
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def generate_training_sequences(num_sequences: int) -> tuple:
    """
    Generate meaningful training sequences with linguistic structure.
    
    Creates simple sentence-like patterns:
    - Subject-verb-object structures
    - Determiner-noun phrases
    - Preposition-noun phrases
    
    Returns:
        sequences: List of token ID sequences
        vocab: Vocabulary mapping tokens to IDs
    """
    # Create a small vocabulary with meaningful words
    vocab = {
        'the': 0, 'a': 1,           # Determiners
        'cat': 2, 'dog': 3, 'bird': 4, 'fish': 5,  # Nouns (animals)
        'mat': 6, 'box': 7, 'tree': 8, 'house': 9,  # Nouns (objects/places)
        'sat': 10, 'ran': 11, 'ate': 12, 'saw': 13,  # Verbs
        'on': 14, 'in': 15, 'by': 16, 'near': 17,   # Prepositions
    }
    
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Define sentence templates (as token ID patterns)
    determiners = [0, 1]  # the, a
    animals = [2, 3, 4, 5]  # cat, dog, bird, fish
    places = [6, 7, 8, 9]   # mat, box, tree, house
    verbs = [10, 11, 12, 13]  # sat, ran, ate, saw
    preps = [14, 15, 16, 17]  # on, in, by, near
    
    sequences = []
    
    # Template 1: "det animal verb prep det place" (e.g., "the cat sat on the mat")
    for _ in range(int(num_sequences * 0.4)):
        seq = torch.tensor([
            determiners[torch.randint(len(determiners), (1,)).item()],
            animals[torch.randint(len(animals), (1,)).item()],
            verbs[torch.randint(len(verbs), (1,)).item()],
            preps[torch.randint(len(preps), (1,)).item()],
            determiners[torch.randint(len(determiners), (1,)).item()],
            places[torch.randint(len(places), (1,)).item()],
        ])
        sequences.append(seq)
    
    # Template 2: "det animal verb det animal" (e.g., "the cat saw a dog")
    for _ in range(int(num_sequences * 0.3)):
        seq = torch.tensor([
            determiners[torch.randint(len(determiners), (1,)).item()],
            animals[torch.randint(len(animals), (1,)).item()],
            verbs[torch.randint(len(verbs), (1,)).item()],
            determiners[torch.randint(len(determiners), (1,)).item()],
            animals[torch.randint(len(animals), (1,)).item()],
            places[torch.randint(len(places), (1,)).item()],  # Added for length
        ])
        sequences.append(seq)
    
    # Template 3: "det animal verb prep det animal" (e.g., "a bird ran near the dog")
    for _ in range(int(num_sequences * 0.3)):
        seq = torch.tensor([
            determiners[torch.randint(len(determiners), (1,)).item()],
            animals[torch.randint(len(animals), (1,)).item()],
            verbs[torch.randint(len(verbs), (1,)).item()],
            preps[torch.randint(len(preps), (1,)).item()],
            determiners[torch.randint(len(determiners), (1,)).item()],
            animals[torch.randint(len(animals), (1,)).item()],
        ])
        sequences.append(seq)
    
    print("\nExample training sequences:")
    for i in range(min(10, len(sequences))):
        tokens = [id_to_token[idx.item()] for idx in sequences[i]]
        print(f"  {' '.join(tokens)}")
    
    return sequences, vocab


def train_model(model, sequences, num_epochs, device, learning_rate=0.001):
    """
    Train the model on next-token prediction.
    
    Args:
        model: The language model
        sequences: List of training sequences
        num_epochs: Number of training epochs
        device: Device to train on
        learning_rate: Learning rate
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(sequences)} sequences for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Learning rate: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(sequences, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for seq in pbar:
            seq = seq.to(device)
            
            # Input: all tokens except last
            # Target: all tokens except first (shifted by 1)
            input_ids = seq[:-1].unsqueeze(0)  # (1, seq_len-1)
            target_ids = seq[1:]  # (seq_len-1,)
            
            # Create causal mask
            seq_len = input_ids.size(1)
            mask = create_causal_mask(seq_len, device)
            
            # Forward pass
            logits = model(input_ids, mask=mask)  # (1, seq_len-1, vocab_size)
            
            # Compute loss
            logits = logits.squeeze(0)  # (seq_len-1, vocab_size)
            loss = criterion(logits, target_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!")


def test_model_on_toy_sequence(model, training_vocab, device):
    """
    Test the trained model on our toy sequence and save attention weights.
    Uses the same vocabulary as training for consistency.
    """
    model.eval()
    
    # Use the training vocabulary
    tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    vocab = training_vocab
    
    print_section("Testing on Toy Sequence")
    print(f"Sequence: {' '.join(tokens)}")
    print(f"Tokens: {tokens}\n")
    
    # Convert to IDs using training vocab
    from shared.utils import tokens_to_ids
    token_ids = tokens_to_ids(tokens, vocab).unsqueeze(0).to(device)
    seq_len = len(tokens)
    
    # Forward pass with attention weights
    with torch.no_grad():
        logits, attention_weights = model(token_ids, return_attention=True)
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1).squeeze(0)
    
    print("Predictions:")
    id_to_token = {v: k for k, v in vocab.items()}
    for i in range(len(tokens)):
        pred_id = predictions[i].item()
        pred_token = id_to_token.get(pred_id, f"<UNK:{pred_id}>")
        print(f"  Position {i} ({tokens[i]}): predicts '{pred_token}'")
    
    # Save trained attention weights
    output_path = DATA_PROCESSED_DIR / "phase1_trained_attention.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'attention_weights': attention_weights,
        'tokens': tokens,
        'logits': logits,
        'predictions': predictions,
        'vocab': vocab
    }, output_path)
    
    print(f"\nSaved trained attention weights to: {output_path}")
    
    return attention_weights


def main():
    """Main training function."""
    # Configuration
    D_MODEL = 64
    D_K = 64
    D_FF = 256
    NUM_SEQUENCES = 1000
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    SEED = 42
    
    print_section("Training Single-Head Self-Attention Model")
    
    # Setup
    set_seed(SEED)
    device = get_device()
    
    # Generate training data (now returns vocab too)
    print_section("Generating Training Data")
    sequences, vocab = generate_training_sequences(NUM_SEQUENCES)
    VOCAB_SIZE = len(vocab)
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    print(f"  Model dimension: {D_MODEL}")
    print(f"  Key/Query dimension: {D_K}")
    print(f"  Feed-forward dimension: {D_FF}")
    print(f"  Training sequences: {len(sequences)}")
    print(f"  Sequence length: 6 (all sequences)")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {device}")
    print(f"  Random seed: {SEED}\n")
    
    # Create model
    model = SimpleLanguageModel(VOCAB_SIZE, D_MODEL, D_K, D_FF).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")
    
    # Train model
    print_section("Training Model")
    train_model(model, sequences, NUM_EPOCHS, device, LEARNING_RATE)
    
    # Test on toy sequence (pass vocab from training)
    attention_weights = test_model_on_toy_sequence(model, vocab, device)
    
    # Save trained model
    model_path = DATA_PROCESSED_DIR / "phase1_trained_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': VOCAB_SIZE,
            'd_model': D_MODEL,
            'd_k': D_K,
            'd_ff': D_FF
        }
    }, model_path)
    print(f"\nSaved trained model to: {model_path}")
    
    print_section("Next Steps")
    print("Run the comparison visualization to see how attention changed:")
    print("  uv run python phase1_single_head/visualize_comparison.py")


if __name__ == "__main__":
    main()

