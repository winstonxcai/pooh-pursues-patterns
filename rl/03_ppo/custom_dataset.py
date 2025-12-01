import torch
from constants import DATASET_NAME, MAX_SEQ_LEN, NUM_SAMPLES, PROMPT_TEMPLATE
from data_processing import build_prompt, clean_text
from datasets import load_dataset
from torch.utils.data import Dataset


class ARCEasyDataset(Dataset):
    """
    ARC-Easy dataset loader for PPO.
    Each item returns (input_ids, attention_mask, label_text).
    PPO later generates actions and computes reward.
    """

    def __init__(self, tokenizer):
        """
        tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

        dataset = load_dataset(DATASET_NAME, split="train")

        if NUM_SAMPLES is not None and NUM_SAMPLES < len(dataset):
            dataset = dataset.select(range(NUM_SAMPLES))

        self.data = list(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        question = clean_text(example["question"])
        labels = example["choices"]["label"]   # ["A","B","C","D"]
        choices = example["choices"]["text"]   # list of 4 strings
        answer_key = example["answerKey"]      # correct answer letter

        # Build formatted prompt
        prompt = build_prompt(
            question=question,
            labels=labels,
            choices=choices,
        )

        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # [seq_len]
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_text": answer_key.strip().upper(),              # e.g., "A"
        }
