import logging
import random

import numpy as np
import torch
from constants import (ALPHA, BATCH_SIZE, BETA, DEVICE, GRAD_ACCUM_STEPS,
                       LEARNING_RATE, LOG_FILE, LOGGING_INTERVAL, MAX_SEQ_LEN,
                       MODEL_NAME, NUM_EPOCHS, NUM_SAMPLES, PROMPT_TEMPLATE,
                       SEED, SPLIT_RATIO, WARMUP_RATIO, R)
from custom_dataset import CosmosQADataset
from dpo_utils import (dpo_loss, kl_divergence, preference_accuracy,
                       sequence_logprobs)
from lora import inject_lora
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)


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

    logger.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Tokenizer lacked pad token; set to eos_token.")

    logger.info("Loading reference model...")
    reference_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    logger.info("Loading and setting up policy model...")
    current_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    current_model.gradient_checkpointing_enable()
    current_model.config.use_cache = False  # save memory during training
    current_model = inject_lora(current_model, R, ALPHA)

    logger.info("Loading Cosmos QA dataset...")
    full_dataset = CosmosQADataset(
        tokenizer=tokenizer,
        max_len=MAX_SEQ_LEN,
        prompt_template=PROMPT_TEMPLATE,
        num_samples=NUM_SAMPLES,
        seed=SEED
    )
    
    total_samples = len(full_dataset)
    test_size = max(1, int((1 - SPLIT_RATIO) * total_samples))
    train_size = total_samples - test_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    
    logger.info("Train samples: %d | Val samples: %d", train_size, test_size)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(current_model.parameters(), lr=LEARNING_RATE)
    total_optimizer_steps = (len(train_loader) * NUM_EPOCHS + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    logger.info("Starting training for %d epochs", NUM_EPOCHS)
    logger.info("Total optimizer steps: %d (warmup: %d)", total_optimizer_steps, warmup_steps)

    steps = 0  # micro steps
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    amp_device = DEVICE.type if DEVICE.type in ("cuda", "mps") else "cpu"
    use_amp = amp_device in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if (amp_device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for epoch in range(NUM_EPOCHS):
        logger.info("Epoch %d/%d -- %d mini-batches", epoch + 1, NUM_EPOCHS, len(train_loader))
        train_iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=False,
        )
        for batch_idx, batch in enumerate(train_iterator):
            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            chosen_labels = batch["chosen_labels"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)
            rejected_labels = batch["rejected_labels"].to(DEVICE)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
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

            loss_to_log = loss.detach()
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            should_step = (steps + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1 == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

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
                    {"kl": f"{kl_div.item():.4f}", "loss": f"{loss_to_log.item():.4f}", "pref_acc": f"{pref_acc.item():.3f}"}
                )
                logger.info(
                    "Step %d: KL Div: %.4f, Loss: %.4f, Pref Acc: %.3f",
                    steps, kl_div.item(), loss_to_log.item(), pref_acc.item()
                )
            steps += 1

        val_metrics = evaluate(current_model, reference_model, val_loader)
        logger.info(
            "Epoch %d/%d - Val DPO Loss: %.4f, Val Pref Acc: %.4f, Val CE: %.4f",
            epoch + 1, NUM_EPOCHS,
            val_metrics['dpo_loss'],
            val_metrics['preference_accuracy'],
            val_metrics['cross_entropy']
        )


if __name__ == "__main__":
    main()
