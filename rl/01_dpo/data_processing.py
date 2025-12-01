import json
import random

from constants import DATA_SAVE_PATH, DATASET_NAME, PROMPT_TEMPLATE
from datasets import load_dataset
from tqdm import tqdm


def build_pair(prompt, chosen, rejected):
    """
    Build a pair of (prompt, chosen, rejected).
    Args:
        prompt: Prompt string
        chosen: Chosen answer string
        rejected: Rejected answer string
    Returns:
        Dictionary with prompt, chosen, and rejected
    """
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    """
    Main function for processing the Cosmos QA dataset.
    """
    data = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)

    with open(DATA_SAVE_PATH, "w") as f:
        skipped_count = 0
        for row in tqdm(data, desc="Processing dataset"):

            context = row["context"]
            question = row["question"]
            label = row["label"]  # 0-3, index of correct answer

            # Check if any answer contains "None of the above" - skip this question if so
            has_none_of_above = False
            for i in range(4):
                answer = row[f"answer{i}"].lower().strip()
                if "none of the above" in answer:
                    has_none_of_above = True
                    break
            
            if has_none_of_above:
                skipped_count += 1
                continue

            # Get the correct answer
            chosen = row[f"answer{label}"]

            # Get all incorrect answer indices
            incorrect_indices = [i for i in range(4) if i != label]
            
            # Randomly choose one incorrect answer
            rejected_idx = random.choice(incorrect_indices)
            rejected = row[f"answer{rejected_idx}"]
            
            # Prompt should not include the answer - it will be concatenated in the dataset
            prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer="")
            pair = build_pair(prompt, chosen, rejected)
            f.write(json.dumps(pair) + "\n")

    print(f"Saved DPO dataset to {DATA_SAVE_PATH}")
    print(f"Skipped {skipped_count} questions containing 'None of the above' answers")

if __name__ == "__main__":
    main()
