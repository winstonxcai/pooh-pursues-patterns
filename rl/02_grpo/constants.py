MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # change if needed
DATA_PATH = "./gsm8k_grpo_real.jsonl"
MAX_SAMPLES = 2000  # set to int to use subset

DEVICE = "cuda"

BATCH_SIZE = 1         # 1 prompt = 4 candidates
NUM_EPOCHS = 3
LR = 3e-4
MAX_LEN = 512

LORA_RANK = 8
LORA_ALPHA = 32

LOG_INTERVAL = 10
TEST_BATCH_SIZE = 8
LOG_FILE = "grpo_training.log"
