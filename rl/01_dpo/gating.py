# gating.py
# Drop-in G1 Gated Attention for LLaMA models
# Works with meta-llama/Llama-3.2-1B-Instruct + LoRA

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel

# ---------------------------------------------------------------------------
# 1. Gated LLaMA Attention (G1 Variant: gate AFTER SDPA)
# ---------------------------------------------------------------------------

class GatedLlamaAttention(LlamaAttention):
    """
    G1: Gate applied AFTER SDPA, BEFORE W_O.
    Gate is head-wise: one scalar per head per token.
    """

    def __init__(self, config):
        super().__init__(config)

        # gate produces (batch, seq, num_heads)
        self.W_gate = nn.Linear(
            config.hidden_size,
            config.num_attention_heads,
            bias=False
        )

        # OPTIONAL: start gate near-neutral (sigmoid(0)=0.5)
        nn.init.zeros_(self.W_gate.weight)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # ---- SAME AS BASE LLAMA ----
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (B, S, H*d) → (B, H, S, d)
        query_states = self._shape(query_states, -1, bsz)
        key_states   = self._shape(key_states,   -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        # ---- SDPA ----
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores = scores / (self.head_dim ** 0.5)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, S, d)

        # reshape back
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        # attn_output shape: (B, S, H*d)

        # -------------------------------------------------------------------
        # NEW: G1 GATING
        # -------------------------------------------------------------------
        # gate_raw: (B, S, num_heads)
        gate_raw = self.W_gate(hidden_states)

        # gate: (B, S, num_heads)
        gate = torch.sigmoid(gate_raw)

        # expand gate to (B, S, H*d)
        gate = gate.repeat_interleave(self.head_dim, dim=-1)

        # apply gate
        attn_output = attn_output * gate

        # ---- SAME AS BASE LLAMA ----
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights

        return attn_output, None


# ---------------------------------------------------------------------------
# 2. Apply Gating to a LLaMA Model
# ---------------------------------------------------------------------------

def apply_gating_to_llama(model):
    """
    Replaces all LlamaAttention layers in a LlamaModel / LlamaForCausalLM
    with the GatedLlamaAttention.
    """

    # LlamaForCausalLM.model → LlamaModel
    if hasattr(model, "model") and isinstance(model.model, LlamaModel):
        llama = model.model
    elif isinstance(model, LlamaModel):
        llama = model
    else:
        raise ValueError("Model must be a LlamaModel or LlamaForCausalLM.")

    config = llama.config

    for i, layer in enumerate(llama.layers):
        layer.self_attn = GatedLlamaAttention(config)

    print("[GATED ATTENTION] Applied G1 gating to all attention layers.")
    return model
