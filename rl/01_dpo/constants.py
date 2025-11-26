import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # or TinyLlama
DATASET_NAME = "Anthropic/hh-rlhf"
NUM_SAMPLES = 1000
SPLIT_RATIO = 0.8
R = 16
ALPHA = 16
BETA = 0.1
LEARNING_RATE = 3e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
LOGGING_INTERVAL = 100
SEED = 42

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")