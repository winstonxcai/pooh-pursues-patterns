from functools import partial

import torch
from constants import (ALPHA, BATCH_SIZE, DATASET_NAME, DEVICE, LEARNING_RATE,
                       LOGGING_INTERVAL, MAX_SEQ_LEN, MODEL_NAME, NUM_EPOCHS,
                       NUM_SAMPLES, SPLIT_RATIO, R)
from datasets import load_dataset as hf_load_dataset
from dpo_utils import compute_logprobs, dpo_loss, kl_divergence
from lora import inject_lora
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_labels(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Clone the token ids and replace padding positions with -100
    so the loss ignores them.
    """
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return labels


def tokenize(example, tokenizer):
    """
    Tokenize the example.
    Format:
    {
        "chosen": ...
        "rejected": ...
    }

    Args:
        example: The example to tokenize.

    Returns:
        The tokenized example.
    """
    chosen_tokens = tokenizer(
        example["chosen"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    rejected_tokens = tokenizer(
        example["rejected"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )

    chosen_ids = chosen_tokens.input_ids.squeeze(0)
    rejected_ids = rejected_tokens.input_ids.squeeze(0)

    chosen_mask = chosen_tokens.attention_mask.squeeze(0)
    rejected_mask = rejected_tokens.attention_mask.squeeze(0)

    chosen_labels = _build_labels(chosen_ids, tokenizer.pad_token_id)
    rejected_labels = _build_labels(rejected_ids, tokenizer.pad_token_id)

    return {
        "chosen_ids": chosen_ids.tolist(),
        "chosen_mask": chosen_mask.tolist(),
        "chosen_labels": chosen_labels.tolist(),
        "rejected_ids": rejected_ids.tolist(),
        "rejected_mask": rejected_mask.tolist(),
        "rejected_labels": rejected_labels.tolist(),
    }


def evaluate(model, data_loader):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            chosen_labels = batch["chosen_labels"].to(DEVICE)

            batch_loss = -compute_logprobs(model, chosen_ids, chosen_mask, chosen_labels)
            total_loss += batch_loss.item()
            total_batches += 1

    if was_training:
        model.train()

    return total_loss / max(total_batches, 1)

def main():
    """
    Main function for training the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reference_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    current_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    current_model = inject_lora(current_model, R, ALPHA)

    optimizer = torch.optim.AdamW(current_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    full_dataset = hf_load_dataset(DATASET_NAME, split="train")
    total_samples = min(NUM_SAMPLES, len(full_dataset)) if NUM_SAMPLES else len(full_dataset)
    full_dataset = full_dataset.select(range(total_samples))

    split_index = int(total_samples * SPLIT_RATIO)
    train_dataset = full_dataset.select(range(split_index))
    val_dataset = full_dataset.select(range(split_index, total_samples))

    tokenize_fn = partial(tokenize, tokenizer=tokenizer)
    train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_fn, remove_columns=val_dataset.column_names)

    tensor_columns = [
        "chosen_ids",
        "chosen_mask",
        "chosen_labels",
        "rejected_ids",
        "rejected_mask",
        "rejected_labels",
    ]
    train_dataset.set_format(type="torch", columns=tensor_columns)
    val_dataset.set_format(type="torch", columns=tensor_columns)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    steps = 0
    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            chosen_labels = batch["chosen_labels"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)
            rejected_labels = batch["rejected_labels"].to(DEVICE)

            computed_logprobs = compute_logprobs(current_model, chosen_ids, chosen_mask, chosen_labels)
            reference_logprobs = compute_logprobs(reference_model, rejected_ids, rejected_mask, rejected_labels)
            loss = dpo_loss(computed_logprobs, reference_logprobs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if steps % LOGGING_INTERVAL == 0:
                with torch.no_grad():
                    kl_div = kl_divergence(
                        current_model,
                        reference_model,
                        {
                            "chosen_ids": chosen_ids,
                            "chosen_mask": chosen_mask,
                            "chosen_labels": chosen_labels,
                            "rejected_ids": rejected_ids,
                            "rejected_mask": rejected_mask,
                            "rejected_labels": rejected_labels,
                        },
                    )
                print(f"Step {steps}: KL Div: {kl_div.item():.4f}, Loss: {loss.item():.4f}")
            steps += 1

        val_loss = evaluate(current_model, val_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()