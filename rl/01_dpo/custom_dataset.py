import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class CosmosQADataset(Dataset):
    """Dataset for Cosmos QA formatted for DPO training."""

    def __init__(self, tokenizer, max_len, prompt_template, num_samples=None, seed=42):
        """
        Initialize the Cosmos QA dataset.

        Args:
            tokenizer: Tokenizer to use for encoding
            max_len: Maximum sequence length
            prompt_template: Template string for prompts (with {context}, {question}, {answer} placeholders)
            num_samples: Optional limit on number of samples to use
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt_template = prompt_template
        
        # Load cosmos_qa dataset
        dataset = load_dataset("allenai/cosmos_qa", split="train")
        
        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        self.data = list(dataset)
        random.seed(seed)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a (chosen, rejected) pair from the dataset.
        
        Returns:
            Dictionary with chosen_ids, chosen_mask, chosen_labels,
            rejected_ids, rejected_mask, rejected_labels
        """
        example = self.data[idx]
        
        # Extract fields
        context = example["context"]
        question = example["question"]
        label = example["label"]  # 0-3, index of correct answer
        
        # Get correct answer
        correct_answer = example[f"answer{label}"]
        
        # Get wrong answers (all answers except the correct one)
        wrong_indices = [i for i in range(4) if i != label]
        wrong_answer = example[f"answer{random.choice(wrong_indices)}"]
        
        # Build prompt base (without answer) for masking
        prompt_base = self.prompt_template.format(
            context=context,
            question=question,
            answer=""  # Empty for prompt base
        ).rstrip()
        
        # Build full sequences with answers
        chosen_text = self.prompt_template.format(
            context=context,
            question=question,
            answer=correct_answer
        )
        rejected_text = self.prompt_template.format(
            context=context,
            question=question,
            answer=wrong_answer
        )
        
        # Tokenize chosen sequence
        chosen_enc = self.tokenizer(
            chosen_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Tokenize rejected sequence
        rejected_enc = self.tokenizer(
            rejected_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        chosen_ids = chosen_enc["input_ids"].squeeze(0)
        rejected_ids = rejected_enc["input_ids"].squeeze(0)
        
        chosen_mask = chosen_enc["attention_mask"].squeeze(0)
        rejected_mask = rejected_enc["attention_mask"].squeeze(0)
        
        # Build labels (mask prompt tokens)
        chosen_labels = chosen_ids.clone()
        rejected_labels = rejected_ids.clone()
        
        prompt_len = len(self.tokenizer(prompt_base)["input_ids"])
        chosen_labels[:prompt_len] = -100
        rejected_labels[:prompt_len] = -100
        
        # Also mask padding tokens
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            chosen_labels[chosen_ids == pad_token_id] = -100
            rejected_labels[rejected_ids == pad_token_id] = -100
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "chosen_labels": chosen_labels,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
            "rejected_labels": rejected_labels,
        }

