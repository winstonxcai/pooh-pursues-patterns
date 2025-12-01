import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "allenai/cosmos_qa"
DATA_SAVE_PATH = "./cosmos_qa_dpo.jsonl"
NUM_SAMPLES = 2000
PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\n\n\nAssistant: {answer}"
USE_GATING = False
LOG_FILE = "dpo_training{}.log".format("_gated" if USE_GATING else "")
SPLIT_RATIO = 0.9
R = 4
ALPHA = 16
LEARNING_RATE = 5e-6
BETA = 0.03
WARMUP_RATIO = 0.1
NUM_EPOCHS = 1
BATCH_SIZE = 4
MAX_SEQ_LEN = 224
LOGGING_INTERVAL = 10
SEED = 42

# Logging format and pattern for parsing
LOG_FORMAT = "[step {step}] loss={loss:.4f} | KL={kl:.4f}"
LOG_PATTERN = r'\[step (\d+)\] loss=([\d.]+) \| KL=([\d.]+)'

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
