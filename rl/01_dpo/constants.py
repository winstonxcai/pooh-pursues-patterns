import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "Anthropic/hh-rlhf"
NUM_SAMPLES = 3000
SPLIT_RATIO = 0.9
R = 32
ALPHA = 32
BETA = 0.1
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1
NUM_EPOCHS = 2
BATCH_SIZE = 2  # per-device microbatch; effective batch is BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 384
LOGGING_INTERVAL = 25
SEED = 42

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
