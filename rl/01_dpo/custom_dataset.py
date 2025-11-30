import json

import torch
from torch.utils.data import Dataset


class CosmosQADataset(Dataset):
    """Dataset for Cosmos QA training."""

    def __init__(self, path, tokenizer, max_len):
        """
        Initialize the Cosmos QA dataset.

        Args:
            path: Path to the dataset file
            tokenizer: Tokenizer to use for encoding
            max_len: Maximum sequence length
        """
        self.data = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_len = max_len
        
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
