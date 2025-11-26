import torch
import torch.nn.functional as F


def sequence_logprobs(model, input_ids, attention_mask, labels=None):
    """
    Compute summed sequence log-probabilities for each example.
    Uses an autoregressive shift (token t predicted at position t-1) and
    ignores positions where labels == -100.

    Returns:
        seq_logprobs: (batch_size,)
        token_counts: (batch_size,) number of tokens contributing to the logprob
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
    token_counts = valid_mask_f.sum(dim=-1).clamp_min(1)

    return seq_logprobs, token_counts


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


def preference_accuracy(policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor) -> torch.Tensor:
    """Compute preference accuracy for a batch."""
    return (policy_chosen_logp > policy_rejected_logp).float().mean()


def kl_divergence(current_model, reference_model, batch):
    """
    Compute the average KL divergence between current and reference models
    over both chosen and rejected sequences.
    """
    def _kl(ids, mask):
        current_logits = current_model(ids, attention_mask=mask).logits
        reference_logits = reference_model(ids, attention_mask=mask).logits

        current_log_probs = F.log_softmax(current_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)

        per_token_kl = F.kl_div(
            current_log_probs,
            reference_probs,
            log_target=False,
            reduction="none",
        ).sum(dim=-1)

        mask_f = mask.to(per_token_kl.dtype)
        masked_kl = (per_token_kl * mask_f).sum()
        normalization = mask_f.sum().clamp_min(1.0)
        return masked_kl / normalization

    chosen_ids = batch["chosen_ids"]
    chosen_mask = batch["chosen_mask"]
    rejected_ids = batch["rejected_ids"]
    rejected_mask = batch["rejected_mask"]

    chosen_kl = _kl(chosen_ids, chosen_mask)
    rejected_kl = _kl(rejected_ids, rejected_mask)
    return 0.5 * (chosen_kl + rejected_kl)
