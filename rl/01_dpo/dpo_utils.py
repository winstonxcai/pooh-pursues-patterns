import torch
import torch.nn.functional as F


def dpo_loss(logp_chosen, logp_rejected, beta=0.1):
    """
    Compute the DPO loss.

    Args:
        logp_chosen: The log probability of the chosen action. (batch_size, )
        logp_rejected: The log probability of the rejected action. (batch_size, )
        beta: The beta parameter for the DPO loss.

    Returns:
        The DPO loss. (batch_size, )
    """
    delta_logp = logp_chosen - logp_rejected
    return -torch.mean(torch.log(torch.sigmoid(beta * delta_logp)))

def compute_logprobs(model, input_ids, attention_mask, labels):
    """
    Compute the log probabilities of the actions. log (labels | input_ids)

    Args:
        model: The model to compute the log probabilities.
        input_ids: The input ids. (batch_size, seq_len)
        attention_mask: The attention mask. (batch_size, seq_len)
        labels: The labels. (batch_size, seq_len)

    Returns:
        The log probabilities. (batch_size, )
    """
    output = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss
    return -loss

def kl_divergence(current_model, reference_model, batch):
    """
    Compute the KL divergence for a batch

    Args:
        current_model: The current model.
        reference_model: The reference model.
        batch: The batch.

    Returns:
        The KL divergence.
    """
    chosen_ids = batch["chosen_ids"]
    chosen_mask = batch["chosen_mask"]
    chosen_labels = batch["chosen_labels"]
    rejected_ids = batch["rejected_ids"]
    rejected_mask = batch["rejected_mask"]
    rejected_labels = batch["rejected_labels"]

    current_logits = current_model(chosen_ids, attention_mask=chosen_mask).logits
    reference_logits = reference_model(chosen_ids, attention_mask=chosen_mask).logits

    current_log_probs = F.log_softmax(current_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)

    return F.kl_div(current_log_probs, reference_probs, log_target=False, reduction="batchmean")