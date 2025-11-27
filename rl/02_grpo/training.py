import torch
from constants import (BATCH_SIZE, DATA_PATH, DEVICE, LOG_INTERVAL, LORA_ALPHA,
                       LORA_RANK, LR, MAX_LEN, MODEL_NAME, NUM_EPOCHS,
                       SAVE_INTERVAL, TEST_BATCH_SIZE)
from custom_dataset import GRPODataset
from grpo_utils import get_answer_log_probs, grpo_loss
from lora import inject_lora
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------
# Setup
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

model = inject_lora(model, LORA_RANK, LORA_ALPHA)
model.train()

dataset = GRPODataset(DATA_PATH, tokenizer, MAX_LEN)
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

# -----------------------------
# Training
# -----------------------------

global_step = 0

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}")
    for batch in tqdm(train_loader):

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

            if global_step % LOG_INTERVAL == 0:
                print(f"step {global_step} | loss: {loss.item()}")

            if global_step % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/grpo_step_{global_step}.pt")

    accuracy = evaluate_accuracy(model, test_loader)
    print(f"Test accuracy after epoch {epoch+1}: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), "checkpoints/final_grpo.pt")
