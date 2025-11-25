import torch
from constants import BATCH_SIZE, DEVICE, SEQ_LEN


def get_batch(data: torch.Tensor, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get a batch of data from the dataset.

    Args:
        data: The dataset to get the batch from.
        split: The split to get the batch from.
    """
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y