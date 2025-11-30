MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # change if needed
DATA_PATH = "./gsm8k_grpo_real.jsonl"
MAX_SAMPLES = 2000  # set to int to use subset

DEVICE = "cuda"

BATCH_SIZE = 4         # 1 prompt = 4 candidates
NUM_EPOCHS = 3
LR = 3e-4
MAX_LEN = 512

LORA_RANK = 8
LORA_ALPHA = 32

LOG_INTERVAL = 10
TEST_BATCH_SIZE = 8
LOG_FILE = "grpo_training.log"
import torch
from advantage import compute_gae
from buffer import PPOBuffer
from custom_dataset import ARCEasyDataset
from ppo_step import ppo_update
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===================== CONFIG ===================== #

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """Question:
{question}

Choices:
{choices}

Answer (A/B/C/D):
"""


# ===================== LOAD MODEL ===================== #

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ===================== DATASET ===================== #

dataset = ARCEasyDataset(
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    prompt_template=PROMPT_TEMPLATE,
    num_samples=5000
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ===================== TRAIN LOOP ===================== #

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    buffer = PPOBuffer()

    for batch in loader:

        input_ids = batch["input_ids"].to(DEVICE)           # [B, seq_len]
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label_text"]                         # B strings

        with torch.no_grad():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        actions = outputs[:, input_ids.shape[1]:]            # [B, gen_len]
        decoded = tokenizer.batch_decode(actions)

        # ================= REWARD ================= #
        rewards = []

        for pred, true in zip(decoded, labels):
            pred = pred.strip().upper()

            if len(pred) == 0:
                rewards.append(-2)
            elif pred[0] == true:
                rewards.append(1)
            else:
                rewards.append(-1)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

        # ================= LOGPROBS + VALUES ================= #

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        last_token = actions[:, 0].unsqueeze(-1)
        action_logprobs = torch.gather(
            log_probs[:, -1, :],
            dim=-1,
            index=last_token
        ).squeeze(-1)

        values = torch.zeros_like(rewards).to(DEVICE)  # placeholder Value Function

        for i in range(len(rewards)):
            buffer.store(
                input_ids[i],
                actions[i],
                action_logprobs[i],
                rewards[i],
                values[i]
            )

    # Get buffer into tensors
    states, actions, old_logprobs, rewards, values = buffer.get()

    # ================= ADVANTAGE ================= #

    advantages, returns = compute_gae(
        rewards.to(DEVICE),
        values.to(DEVICE)
    )

    # ================= PPO UPDATE ================= #

    stats = ppo_update(
        model=model,
        optimizer=optimizer,
        states=states.to(DEVICE),
        actions=actions.to(DEVICE),
        old_logprobs=old_logprobs.to(DEVICE),
        advantages=advantages.to(DEVICE),
        returns=returns.to(DEVICE)
    )

    print(stats)
