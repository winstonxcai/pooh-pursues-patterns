import torch
from buffer import PPOBuffer
from constants import (COLLECT_BATCH_SIZE, DEVICE, DTYPE, ENTROPY_COEF,
                       GRADIENT_CLIP, LEARNING_RATE, LORA_ALPHA, LORA_R,
                       MAX_NEW_TOKENS, MINIBATCH_SIZE, MODEL_NAME, PPO_EPOCHS,
                       PPO_ITERATIONS, VALUE_COEF, WEIGHT_DECAY)
from custom_dataset import ARCEasyDataset
from data_processing import compute_reward, decode_actions
from lora import inject_lora
from ppo_utils import ppo_loss, split_minibatches, write_log
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Load Model + LoRA
# ============================================================

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto"
    )

    # Inject LoRA modifications
    model = inject_lora(model, r=LORA_R, alpha=LORA_ALPHA)

    # Simple scalar value head on top of hidden size
    model.value_head = torch.nn.Linear(model.config.hidden_size, 1, device=DEVICE, dtype=DTYPE)

    return tokenizer, model


# ============================================================
# PPO Training Loop
# ============================================================

def train():
    tokenizer, model = load_model()
    model.train()
    model.to(DEVICE)

    # Old policy (for rollouts)
    old_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    old_model = inject_lora(old_model, r=LORA_R, alpha=LORA_ALPHA)
    old_model.value_head = torch.nn.Linear(old_model.config.hidden_size, 1, device=DEVICE, dtype=DTYPE)
    old_model.load_state_dict(model.state_dict(), strict=False)
    old_model.value_head.load_state_dict(model.value_head.state_dict())
    old_model.eval()

    dataset = ARCEasyDataset(tokenizer)
    loader = DataLoader(dataset, batch_size=COLLECT_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    step = 0

    loader_iter = iter(loader)

    for iteration in range(PPO_ITERATIONS):

        # Collect rollout batch
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        gold_labels = batch["label_text"]

        buffer = PPOBuffer()

        # ============================================
        # (1) Run π_old to generate actions
        # ============================================
        with torch.no_grad():
            outputs = old_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        actions = outputs[:, input_ids.shape[1]:]  # only new tokens
        action_attention = (actions != tokenizer.pad_token_id).long().to(DEVICE)
        decoded = decode_actions(tokenizer, actions)

        # ============================================
        # (2) Compute rewards for each sample
        # ============================================
        rewards = []
        for pred, gold in zip(decoded, gold_labels):
            reward = compute_reward(pred, gold)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

        # ============================================
        # (3) Compute old logprobs and values (per sample, not across batch)
        # ============================================
        with torch.no_grad():
            full_input = torch.cat([input_ids, actions], dim=1)
            full_attention = torch.cat([attention_mask, action_attention], dim=1)

            outputs_old = old_model(
                full_input,
                attention_mask=full_attention,
                output_hidden_states=True
            )
            log_probs = torch.log_softmax(outputs_old.logits[:, :-1, :], dim=-1)
            labels = full_input[:, 1:]

            prompt_lengths = attention_mask.sum(dim=1)
            action_lengths = action_attention.sum(dim=1)

            token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for i in range(mask.size(0)):
                start = max(prompt_lengths[i].item() - 1, 0)
                end = start + action_lengths[i].item()
                mask[i, start:end] = True

            old_logprobs = (token_logprobs * mask).sum(dim=1).float()

            # Value from hidden state before first action token
            hidden = outputs_old.hidden_states[-1]
            idx = (prompt_lengths - 1).clamp(min=0)
            value_states = hidden[torch.arange(hidden.size(0)), idx]
            values = old_model.value_head(value_states).squeeze(-1).float()

        # Store rollout
        for i in range(len(rewards)):
            buffer.store(
                input_ids[i],
                attention_mask[i],
                actions[i],
                action_attention[i],
                old_logprobs[i],
                rewards[i],
                values[i]
            )

        # Convert to tensors
        states, attn_masks, actions, action_masks, old_logprobs, rewards, values = buffer.get()

        # Ensure everything is on the training device
        states = states.to(DEVICE)
        attn_masks = attn_masks.to(DEVICE)
        actions = actions.to(DEVICE)
        action_masks = action_masks.to(DEVICE)
        old_logprobs = old_logprobs.to(DEVICE)
        rewards = rewards.to(DEVICE)
        values = values.to(DEVICE)

        # ============================================
        # (4) Compute advantages (single-step episodes)
        # ============================================
        advantages = (rewards - values).float()
        returns = rewards.float()

        # ============================================
        # (5) PPO Optimization
        # ============================================
        for epoch in range(PPO_EPOCHS):
            minibatches = split_minibatches(
                states, attn_masks, actions, action_masks, old_logprobs, advantages, returns,
                minibatch_size=MINIBATCH_SIZE
            )

            for mb_states, mb_attn, mb_actions, mb_action_masks, mb_old_logprobs, mb_adv, mb_returns in minibatches:

                # Forward pass for π_new on prompt + actions
                full_input = torch.cat([mb_states.to(DEVICE), mb_actions.to(DEVICE)], dim=1)
                full_attention = torch.cat([mb_attn.to(DEVICE), mb_action_masks.to(DEVICE)], dim=1)

                outputs_new = model(
                    full_input,
                    attention_mask=full_attention,
                    output_hidden_states=True
                )
                log_probs = torch.log_softmax(outputs_new.logits[:, :-1, :], dim=-1)

                labels = full_input[:, 1:]
                prompt_lengths = mb_attn.sum(dim=1)
                action_lengths = mb_action_masks.sum(dim=1)

                token_logprobs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for i in range(mask.size(0)):
                    start = max(prompt_lengths[i].item() - 1, 0)
                    end = start + action_lengths[i].item()
                    mask[i, start:end] = True

                new_logprobs = (token_logprobs * mask).sum(dim=1).float()

                # Value baseline from hidden state before first action
                hidden = outputs_new.hidden_states[-1]
                idx = (prompt_lengths - 1).clamp(min=0)
                value_states = hidden[torch.arange(hidden.size(0)), idx]
                new_values = model.value_head(value_states).squeeze(-1).float()

                # PPO Loss
                loss, pol_loss, val_loss, entropy, kl = ppo_loss(
                    new_logprobs,
                    mb_old_logprobs.to(DEVICE),
                    mb_adv.to(DEVICE),
                    new_values,
                    mb_returns.to(DEVICE)
                )

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()

                step += 1

        # ============================================
        # (6) Log iteration metrics
        # ============================================
        mean_reward = rewards.mean().item()
        log_line = write_log(
            iteration,
            reward=mean_reward,
            kl=kl.item(),
            entropy=entropy.item(),
            policy_loss=pol_loss.item(),
            value_loss=val_loss.item()
        )
        print(log_line)

        # ============================================
        # (7) Update θ_old ← θ_new
        # ============================================
        old_model.load_state_dict(model.state_dict())


if __name__ == "__main__":
    train()
