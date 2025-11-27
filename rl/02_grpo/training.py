import logging

import torch
from constants import (BATCH_SIZE, DATA_PATH, DEVICE, LOG_FILE, LOG_INTERVAL,
                       LORA_ALPHA, LORA_RANK, LR, MAX_LEN, MAX_SAMPLES,
                       MODEL_NAME, NUM_EPOCHS, TEST_BATCH_SIZE)
from custom_dataset import GRPODataset
from grpo_utils import get_answer_log_probs, grpo_loss
from lora import inject_lora
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# -----------------------------
# Setup
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Tokenizer lacked pad token; set to eos_token.")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

model = inject_lora(model, LORA_RANK, LORA_ALPHA)
model.train()

logger.info("Loading dataset from %s", DATA_PATH)
dataset = GRPODataset(DATA_PATH, tokenizer, MAX_LEN)

if MAX_SAMPLES is not None:
    original_len = len(dataset)
    subset_len = min(MAX_SAMPLES, original_len)
    generator = torch.Generator().manual_seed(42)
    subset_indices = torch.randperm(original_len, generator=generator)[:subset_len].tolist()
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    logger.info("Using subset: %d/%d samples", subset_len, original_len)

test_size = max(1, int(0.1 * len(dataset)))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)
collate = lambda batch: batch
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
logger.info("Train samples: %d | Test samples: %d", train_size, test_size)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR
)

def evaluate_accuracy(model, data_loader):
    """Compute accuracy on held-out data."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            for sample in batch:
                candidate_scores = []

                for (input_ids, mask, labels) in sample:
                    input_ids = input_ids.unsqueeze(0).to(DEVICE)
                    mask = mask.unsqueeze(0).to(DEVICE)
                    labels = labels.unsqueeze(0).to(DEVICE)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=mask
                    )

                    shift_logits = outputs.logits[:, :-1, :]
                    shift_labels = labels[:, 1:]

                    score = get_answer_log_probs(shift_logits, shift_labels)
                    candidate_scores.append(score.squeeze(0))

                scores = torch.stack(candidate_scores)
                predicted_idx = torch.argmax(scores).item()

                if predicted_idx == 0:
                    correct += 1
                total += 1

    if was_training:
        model.train()

    return correct / max(total, 1)


logger.info("Computing baseline accuracy on test split...")
baseline_accuracy = evaluate_accuracy(model, test_loader)
logger.info("Baseline test accuracy: %.2f%%", baseline_accuracy * 100)

# -----------------------------
# Training
# -----------------------------

global_step = 0
running_loss = 0.0
running_count = 0
logger.info("Starting training for %d epochs", NUM_EPOCHS)

for epoch in range(NUM_EPOCHS):

    logger.info("Epoch %d/%d -- %d mini-batches", epoch + 1, NUM_EPOCHS, len(train_loader))
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):

        for sample in batch:
            batch_scores = []

            for (input_ids, mask, labels) in sample:

                input_ids = input_ids.unsqueeze(0).to(DEVICE)
                mask = mask.unsqueeze(0).to(DEVICE)
                labels = labels.unsqueeze(0).to(DEVICE)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=mask
                )

                shift_logits = outputs.logits[:, :-1, :]
                shift_labels = labels[:, 1:]

                score = get_answer_log_probs(shift_logits, shift_labels)

                batch_scores.append(score)

            scores = torch.stack(batch_scores)

            loss = grpo_loss(scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            running_count += 1

            if global_step % LOG_INTERVAL == 0:
                avg_loss = running_loss / max(running_count, 1)
                logger.info("step %d | avg_loss (last %d steps): %.4f", global_step, running_count, avg_loss)
                running_loss = 0.0
                running_count = 0

    accuracy = evaluate_accuracy(model, test_loader)
    logger.info("Test accuracy after epoch %d: %.2f%%", epoch + 1, accuracy * 100)

logger.info("Evaluating final model on test split...")
final_accuracy = evaluate_accuracy(model, test_loader)
logger.info(
    "Final test accuracy: %.2f%% (Δ vs. baseline: %+0.2f%%)",
    final_accuracy * 100,
    (final_accuracy - baseline_accuracy) * 100,
)

