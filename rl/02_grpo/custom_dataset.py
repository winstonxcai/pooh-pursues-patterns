import json

import torch
from torch.utils.data import Dataset


class GRPODataset(Dataset):

    def __init__(self, path, tokenizer, max_len):
        self.data = [json.loads(line) for line in open(path)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data[idx]
        prompt = row["prompt"]
        candidates = row["candidates"]

        inputs = []

        for c in candidates:
            text = prompt + "\nAssistant:\n" + c

            enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

            # (1, seq_len)
            input_ids = enc["input_ids"].squeeze(0)
            # (1, seq_len)
            attention_mask = enc["attention_mask"].squeeze(0)

            labels = input_ids.clone()
            prompt_len = len(self.tokenizer(prompt)["input_ids"])
            # mask prompt tokens
            labels[:prompt_len] = -100

            inputs.append((input_ids, attention_mask, labels))

        return inputs
