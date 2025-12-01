# ===========================
# Global Constants for PPO Project
# ===========================

import torch

# ---------------------------
# Model & Tokenizer Settings
# ---------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Prefer bf16 on GPU (supported on L40S), fall back to fp32 on CPU
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN = 256
MAX_NEW_TOKENS = 4    # ARC answers are 1–2 tokens


# ---------------------------
# PPO Hyperparameters
# ---------------------------

PPO_ITERATIONS = 200       # Outer-loop iterations
PPO_EPOCHS = 4             # K epochs per iteration
MINIBATCH_SIZE = 64        # M: minibatch size
COLLECT_BATCH_SIZE = 256   # N: rollout batch size

GAMMA = 0.99               # Discount factor
LAMBDA = 0.95              # GAE λ
CLIP_EPS = 0.2             # PPO clipping epsilon

VALUE_COEF = 0.5           # c1
ENTROPY_COEF = 0.01        # c2

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0

GRADIENT_CLIP = 1.0


# ---------------------------
# LoRA Settings (Always ON)
# ---------------------------

LORA_R = 8
LORA_ALPHA = 32
# Targets loaded inside lora.py: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj


# ---------------------------
# Dataset Settings
# ---------------------------

DATASET_NAME = "allenai/arc_easy"
NUM_SAMPLES = 5000

PROMPT_TEMPLATE = """Question:
{question}

Choices:
{choices}

Answer (A/B/C/D):
"""


# ---------------------------
# Logging
# ---------------------------

LOG_FILE = "ppo_training.log"
# Example:
# [2025-12-01 21:00:10] iter=12 | reward=0.4200 | kl=0.013205 | entropy=1.8150 | policy=0.2140 | value=0.3350
LOG_FORMAT = (
    "[{timestamp}] iter={iteration} | "
    "reward={reward:.4f} | "
    "kl={kl:.6f} | "
    "entropy={entropy:.4f} | "
    "policy={policy_loss:.4f} | "
    "value={value_loss:.4f}"
)

# Regex to parse the above log lines if needed
LOG_PATTERN = (
    r"\[(.*?)\] iter=(\d+) \| "
    r"reward=([\d.\-]+) \| "
    r"kl=([\d.\-]+) \| "
    r"entropy=([\d.\-]+) \| "
    r"policy=([\d.\-]+) \| "
    r"value=([\d.\-]+)"
)
EVAL_EVERY = 10   # Evaluate ARC accuracy every X iterations

SEED = 42
