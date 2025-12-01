import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "allenai/cosmos_qa"
DATA_SAVE_PATH = "./cosmos_qa_dpo.jsonl"
NUM_SAMPLES = 2000
PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\n\n\nAssistant: {answer}"
LOG_FILE = "dpo_training.log"
SPLIT_RATIO = 0.9
R = 32
ALPHA = 32
BETA = 0.7
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
NUM_EPOCHS = 2
BATCH_SIZE = 2 
MAX_SEQ_LEN = 384
LOGGING_INTERVAL = 25
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
