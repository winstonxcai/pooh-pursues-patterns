import re
from datetime import datetime

import torch
from constants import PROMPT_TEMPLATE

# ============================================================
# ARC-Easy Text Cleaning & Formatting
# ============================================================

def clean_text(text: str) -> str:
    """Basic cleanup for ARC-Easy fields."""
    return (
        text.replace("\n", " ")
            .replace("\t", " ")
            .strip()
    )


def format_choices(choice_labels, choice_texts):
    """
    Converts choices into:

        A) option1
        B) option2
        C) option3
        D) option4

    Returns a single string ready to fit into PROMPT_TEMPLATE.
    """
    out = []
    for lbl, txt in zip(choice_labels, choice_texts):
        lbl = lbl.strip().upper()
        txt = clean_text(txt)
        out.append(f"{lbl}) {txt}")
    return "\n".join(out)


def build_prompt(question: str, labels: list, choices: list):
    """Fills the PROMPT_TEMPLATE for PPO rollout."""
    formatted_choices = format_choices(labels, choices)

    return PROMPT_TEMPLATE.format(
        question=clean_text(question),
        choices=formatted_choices
    ).strip()


# ============================================================
# Label Mapping / Reward Helpers
# ============================================================

ARC_LABELS = ["A", "B", "C", "D"]
ARC_TO_INDEX = {lbl: i for i, lbl in enumerate(ARC_LABELS)}


def normalize_pred(pred_text: str):
    """
    Normalize model prediction:
    - uppercase
    - use only first A/B/C/D character
    """
    pred_text = pred_text.strip().upper()

    if len(pred_text) == 0:
        return None

    first = pred_text[0]
    return first if first in ARC_LABELS else None


def compute_reward(pred: str, gold: str):
    """
    Reward function for ARC-Easy PPO.

        +1 = correct
        -1 = incorrect
        -2 = invalid output
    """
    pred_norm = normalize_pred(pred)

    if pred_norm is None:
        return -2

    return 1 if pred_norm == gold else -1


# ============================================================
# Decoding Helpers
# ============================================================

def decode_actions(tokenizer, generated_ids):
    """
    Convert model-generated token IDs into raw text.
    """
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
