import random
from functools import partial

import torch
import numpy as np
from constants import (ALPHA, BATCH_SIZE, BETA, DATASET_NAME, DEVICE, LEARNING_RATE,
                       LOGGING_INTERVAL, MAX_SEQ_LEN, MODEL_NAME, NUM_EPOCHS,
                       NUM_SAMPLES, SEED, SPLIT_RATIO, WARMUP_RATIO, R)
from datasets import load_dataset as hf_load_dataset
from dpo_utils import dpo_loss, kl_divergence, preference_accuracy, sequence_logprobs
from lora import inject_lora
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)


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


def evaluate(policy_model, reference_model, data_loader):
    was_training = policy_model.training
    policy_model.eval()
    reference_model.eval()

    total_examples = 0
    total_dpo_loss = 0.0
    total_pref_acc = 0.0
    total_ce_nll = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            desc="Evaluating",
            leave=False,
        ):
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            chosen_labels = batch["chosen_labels"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)
            rejected_labels = batch["rejected_labels"].to(DEVICE)

            policy_chosen_logp, chosen_token_counts = sequence_logprobs(
                policy_model, chosen_ids, chosen_mask, chosen_labels
            )
            policy_rejected_logp, _ = sequence_logprobs(
                policy_model, rejected_ids, rejected_mask, rejected_labels
            )
            reference_chosen_logp, _ = sequence_logprobs(
                reference_model, chosen_ids, chosen_mask, chosen_labels
            )
            reference_rejected_logp, _ = sequence_logprobs(
                reference_model, rejected_ids, rejected_mask, rejected_labels
            )

            batch_dpo_loss = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                reference_chosen_logp,
                reference_rejected_logp,
                beta=BETA,
            )
            batch_pref_acc = preference_accuracy(policy_chosen_logp, policy_rejected_logp)

            batch_size = chosen_ids.size(0)
            total_examples += batch_size
            total_dpo_loss += batch_dpo_loss.item() * batch_size
            total_pref_acc += batch_pref_acc.item() * batch_size
            total_ce_nll += (-policy_chosen_logp).sum().item()
            total_tokens += chosen_token_counts.sum().item()

    if was_training:
        policy_model.train()

    return {
        "dpo_loss": total_dpo_loss / max(total_examples, 1),
        "preference_accuracy": total_pref_acc / max(total_examples, 1),
        "cross_entropy": total_ce_nll / max(total_tokens, 1),
    }

def main():
    """
    Main function for training the model.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reference_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    current_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    current_model = inject_lora(current_model, R, ALPHA)

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

    optimizer = torch.optim.AdamW(current_model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    steps = 0
    for epoch in range(NUM_EPOCHS):
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=False,
        )
        for batch in train_iterator:
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            chosen_labels = batch["chosen_labels"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)
            rejected_labels = batch["rejected_labels"].to(DEVICE)

            policy_chosen_logp, _ = sequence_logprobs(
                current_model, chosen_ids, chosen_mask, chosen_labels
            )
            policy_rejected_logp, _ = sequence_logprobs(
                current_model, rejected_ids, rejected_mask, rejected_labels
            )
            with torch.no_grad():
                reference_chosen_logp, _ = sequence_logprobs(
                    reference_model, chosen_ids, chosen_mask, chosen_labels
                )
                reference_rejected_logp, _ = sequence_logprobs(
                    reference_model, rejected_ids, rejected_mask, rejected_labels
                )
            loss = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                reference_chosen_logp,
                reference_rejected_logp,
                beta=BETA,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if steps % LOGGING_INTERVAL == 0:
                with torch.no_grad():
                    was_training = current_model.training
                    current_model.eval()
                    kl_div = kl_divergence(
                        current_model,
                        reference_model,
                        {
                            "chosen_ids": chosen_ids,
                            "chosen_mask": chosen_mask,
                            "rejected_ids": rejected_ids,
                            "rejected_mask": rejected_mask,
                        },
                    )
                    if was_training:
                        current_model.train()
                pref_acc = preference_accuracy(policy_chosen_logp, policy_rejected_logp)
                train_iterator.set_postfix(
                    {"kl": f"{kl_div.item():.4f}", "loss": f"{loss.item():.4f}", "pref_acc": f"{pref_acc.item():.3f}"}
                )
                print(
                    f"Step {steps}: KL Div: {kl_div.item():.4f}, Loss: {loss.item():.4f}, Pref Acc: {pref_acc.item():.3f}"
                )
            steps += 1

        val_metrics = evaluate(current_model, reference_model, val_loader)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Val DPO Loss: {val_metrics['dpo_loss']:.4f}, "
            f"Val Pref Acc: {val_metrics['preference_accuracy']:.4f}, "
            f"Val CE: {val_metrics['cross_entropy']:.4f}"
        )


if __name__ == "__main__":
    main()
