import torch


class PPOBuffer:
    """
    Stores rollout data for one PPO iteration.
    Each entry corresponds to a *single prompt* in ARC-Easy PPO.
    """

    def __init__(self):
        self.states = []
        self.attention_masks = []
        self.actions = []
        self.action_masks = []
        self.old_logprobs = []
        self.rewards = []
        self.values = []

    def store(self, state, attention_mask, action, action_mask, old_logprob, reward, value):
        """
        state: input_ids tensor [seq_len]
        attention_mask: mask for the state [seq_len]
        action: generated token(s) [gen_len] (padded)
        action_mask: mask for generated tokens [gen_len]
        old_logprob: scalar log π_old(a|s)
        reward: scalar float
        value: scalar float V(s)
        """
        self.states.append(state)
        self.attention_masks.append(attention_mask)
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.old_logprobs.append(old_logprob)
        self.rewards.append(reward)
        self.values.append(value)

    def get(self):
        """
        Converts stored lists to tensors.
        Each tensor has shape [N, ...] where N is number of rollout samples.
        """
        states = torch.stack(self.states)
        attention_masks = torch.stack(self.attention_masks)
        actions = torch.stack(self.actions)
        action_masks = torch.stack(self.action_masks)
        old_logprobs = torch.stack(self.old_logprobs)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)

        return states, attention_masks, actions, action_masks, old_logprobs, rewards, values
