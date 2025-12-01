import json
import random
import warnings

import torch
from constants import DATA_SAVE_PATH, MAX_SEQ_LEN, MODEL_NAME, NUM_SAMPLES
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CosmosQADataset(Dataset):
    """Dataset for Cosmos QA training."""

    def __init__(self, path, tokenizer, max_len, num_samples):
        """
        Initialize the Cosmos QA dataset.

        Args:
            path: Path to the dataset file
            tokenizer: Tokenizer to use for encoding
            max_len: Maximum sequence length
            num_samples: Number of samples to use
        """
        # randomly sample num_samples from the dataset
        all_data = [json.loads(l) for l in open(path)]
        self.data = random.sample(all_data, num_samples)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_samples = num_samples
        
        # Verify that max_len is appropriate (at least 90% should fit)
        self._verify_max_len()
        
    def _verify_max_len(self):
        """
        Check that at least 90% of tokenized sequences fit within max_len.
        """
        lengths = []
        for row in self.data:
            # Check both chosen and rejected sequences
            chosen_text = row["prompt"] + " " + row["chosen"]
            rejected_text = row["prompt"] + " " + row["rejected"]
            
            chosen_tokens = self.tokenizer(chosen_text, add_special_tokens=True)["input_ids"]
            rejected_tokens = self.tokenizer(rejected_text, add_special_tokens=True)["input_ids"]
            
            lengths.append(len(chosen_tokens))
            lengths.append(len(rejected_tokens))
        
        below_max = sum(1 for length in lengths if length <= self.max_len)
        percentage = (below_max / len(lengths)) * 100
        
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)
        
        print(f"Tokenized sequence length stats:")
        print(f"  Max length: {max_length}")
        print(f"  Average length: {avg_length:.1f}")
        print(f"  Samples within max_len ({self.max_len}): {percentage:.1f}%")
        
        if percentage < 90:
            warnings.warn(
                f"Only {percentage:.1f}% of samples fit within max_len={self.max_len}. "
                f"Consider increasing max_len to at least {max_length} to cover 100% of samples, "
                f"or at least {int(avg_length * 1.5)} to cover ~90%.",
                UserWarning
            )
        else:
            print(f"✓ max_len={self.max_len} is appropriate ({percentage:.1f}% of samples fit)")
    
    def tokenize(self, text):
        """
        Tokenize a text string.

        Args:
            text: Text string to tokenize
        Returns:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
        """
        out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        """
        Get an item from the dataset.

        Args:
            i: Index of the item
        Returns:
            Dictionary with chosen_ids, chosen_mask, rejected_ids, and rejected_mask
        """
        row = self.data[i]
        p = row["prompt"]
        c = row["chosen"]
        r = row["rejected"]

        chosen_ids, chosen_mask = self.tokenize(p + " " + c)
        rejected_ids, rejected_mask = self.tokenize(p + " " + r)

        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = CosmosQADataset(DATA_SAVE_PATH, tokenizer, MAX_SEQ_LEN, NUM_SAMPLES)