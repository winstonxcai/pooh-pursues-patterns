
import torch
import torch.nn.functional as F


def grpo_loss(scores):
    """
    scores: tensor of shape (4,)
    s1 > s2 > s3 > s4
    """
    loss = 0.0
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            loss += -F.logsigmoid(scores[i] - scores[j])
    return loss


def get_answer_log_probs(logits, labels):
    """
    Get the log probabilities of the answer tokens.

    Args:
        logits: The logits of the model. (batch_size, seq_len - 1, vocab_size)
        labels: The token ids of the answer tokens. (batch_size, seq_len - 1)

    Returns:
        The log probabilities of the answer tokens. (batch_size,)
    """
    log_probs = F.log_softmax(logits, dim=-1)

    next_token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    mask = labels != -100

    return (next_token_log_probs * mask).sum(dim=-1)




