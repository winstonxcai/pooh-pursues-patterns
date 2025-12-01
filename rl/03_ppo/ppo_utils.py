from datetime import datetime

import torch
import torch.nn.functional as F
from constants import (CLIP_EPS, ENTROPY_COEF, GAMMA, LAMBDA, LOG_FILE,
                       LOG_FORMAT, MINIBATCH_SIZE, VALUE_COEF)

# ============================================================
# Advantage: Generalized Advantage Estimation (GAE)
# ============================================================

def compute_gae(rewards, values):
    """
    rewards: [N]
    values:  [N]
    Returns:
        advantages: [N]
        returns:    [N]
    """

    N = len(rewards)

    # Append final value (V(s_{T+1})) as 0 because ARC-Easy episodes terminate after one action
    values_ext = torch.cat([values, torch.tensor([0.0], device=values.device)])

    advantages = torch.zeros(N, device=values.device)
    gae = 0

    for t in reversed(range(N)):
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + GAMMA * values_ext[t + 1] - values_ext[t]

        # A_t = δ_t + γλ * A_{t+1}
        gae = delta + GAMMA * LAMBDA * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


# ============================================================
# PPO Loss Functions (Clipped Surrogate)
# ============================================================

def ppo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    new_values,
    returns
):
    """
    Computes:
    - PPO clipped policy loss
    - value loss
    - entropy bonus
    - KL divergence
    """

    # Ratio r_t = π_new / π_old
    ratio = torch.exp(new_logprobs - old_logprobs)

    # Clipped surrogate objective
    clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)

    policy_loss_1 = ratio * advantages
    policy_loss_2 = clipped_ratio * advantages
    policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2))

    # Value loss
    value_loss = VALUE_COEF * F.mse_loss(new_values, returns)

    # Entropy
    entropy = ENTROPY_COEF * (-torch.mean(torch.exp(new_logprobs) * new_logprobs))

    # KL (for logging)
    kl = torch.mean(old_logprobs - new_logprobs)

    # Total loss
    total_loss = policy_loss + value_loss - entropy

    return total_loss, policy_loss, value_loss, entropy, kl


# ============================================================
# Minibatch Splitter (for PPO epochs)
# ============================================================

def split_minibatches(states, attention_masks, actions, action_masks, old_logprobs, advantages, returns, minibatch_size=None):
    """
    Randomly shuffles indices and yields minibatches.
    """
    if minibatch_size is None:
        minibatch_size = MINIBATCH_SIZE

    B = states.size(0)
    indices = torch.randperm(B, device=states.device)

    for start in range(0, B, minibatch_size):
        end = start + minibatch_size
        idx = indices[start:end]

        yield (
            states[idx],
            attention_masks[idx],
            actions[idx],
            action_masks[idx],
            old_logprobs[idx],
            advantages[idx],
            returns[idx],
        )


# ============================================================
# Logging Helper (Matches your exact requested format)
# ============================================================

def write_log(step, reward, kl, entropy, policy_loss, value_loss):
    """
    Writes a line to ppo_training.log in your EXACT chosen log format:

    Example:
    [2025-12-01 21:00:10] iter=12 | reward=0.4200 | kl=0.013205 | entropy=1.8150 | policy=0.2140 | value=0.3350
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = LOG_FORMAT.format(
        timestamp=timestamp,
        iteration=step,
        reward=reward,
        kl=kl,
        entropy=entropy,
        policy_loss=policy_loss,
        value_loss=value_loss
    )

    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    return line
