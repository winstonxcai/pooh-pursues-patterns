import torch
import torch.nn.functional as F


def sequence_logprobs(model, input_ids, attention_mask, labels=None):
    """
    Compute summed sequence log-probabilities for each example.
    Uses an autoregressive shift (token t predicted at position t-1) and
    ignores positions where labels == -100.

    Returns:
        seq_logprobs: (batch_size,) summed log-probabilities for each sequence
    """
    if labels is None:
        labels = input_ids

    outputs = model(input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:].bool()
    valid_mask = (shift_labels != -100) & shift_mask

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_logprobs = log_probs[:, :-1].gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    valid_mask_f = valid_mask.to(token_logprobs.dtype)
    token_logprobs = token_logprobs * valid_mask_f

    seq_logprobs = token_logprobs.sum(dim=-1)

    return seq_logprobs


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Compute the DPO loss:
    -log sigmoid(beta * [(pi(y+|x) - pi(y-|x)) - (pi_ref(y+|x) - pi_ref(y-|x))])
    """
    delta = (policy_chosen_logp - policy_rejected_logp) - (reference_chosen_logp - reference_rejected_logp)
    return -torch.log(torch.sigmoid(beta * delta)).mean()

def kl_divergence(policy_logits, ref_logits, mask):
    """
    policy_logits: (B, T, V)
    ref_logits:    (B, T, V)
    mask:          (B, T)
    returns: scalar KL
    """

    log_p = F.log_softmax(policy_logits, dim=-1)    # (B, T, V)
    log_q = F.log_softmax(ref_logits, dim=-1)       # (B, T, V)
    p = log_p.exp()                                 # (B, T, V)

    # tokenwise KL : sum over vocabulary v
    kl = (p * (log_p - log_q)).sum(dim=-1)          # (B, T)

    kl = kl * mask
    return kl.sum() / mask.sum()