from pathlib import Path

import torch

# Get the root directory (3 levels up from this file)
_ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = _ROOT_DIR / "data" / "raw" / "04_transformer_block"
MODEL_PATH = _ROOT_DIR / "models" / "checkpoints" / "04_transformer_block"
VISUALIZATIONS_PATH = _ROOT_DIR / "visualizations" / "plots" / "04_transformer_block"

# Create directories if they don't exist
MODEL_PATH.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_PATH.mkdir(parents=True, exist_ok=True)
DATA_PATH.mkdir(parents=True, exist_ok=True)


# ML Constants
SEQ_LEN = 64
BATCH_SIZE = 64
D_MODEL = 384
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.2
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_ITERS = 200
EVAL_INTERVAL = 500
SEED = 42

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")