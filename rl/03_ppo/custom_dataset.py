import random

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class PPODataset(Dataset):
    """Dataset for PPO training."""

    def __init__(self, tokenizer, seq_len, prompt_template, num_samples=None):
        """
        Initialize the PPO dataset.

        Args:
            tokenizer: Tokenizer to use for encoding
            seq_len: Sequence length
            prompt_template: Template string for prompts
            num_samples: Optional limit on number of samples
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prompt_template = prompt_template
        
        dataset = load_dataset("allenai/arc_easy", split="train")
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        self.data = list(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return:
            input_ids     : [max_len]
            attention_mask: [max_len]
            label_text    : correct answer letter (A/B/C/D)
        """

        example = self.data[idx]

        question = example["question"]
        choices = example["choices"]["text"]
        labels = example["choices"]["label"]

        answer_key = example["answerKey"]

        choice_text = ""
        for lbl, txt in zip(labels, choices):
            choice_text += f"{lbl}) {txt}\n"

        prompt = self.prompt_template.format(
            question=question,
            choices=choice_text
        )

        enc = self.tokenizer(
            prompt,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),          # [max_len]
            "attention_mask": enc["attention_mask"].squeeze(0),# [max_len]
            "label_text": answer_key                            # e.g. "A"
        }

