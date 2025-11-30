import json

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
        for row in tqdm(data, desc="Processing dataset"):

            context = row["context"]
            question = row["question"]
            label = row["label"]  # 0-3, index of correct answer

            # Get the correct answer
            chosen = row[f"answer{label}"]

            # Generate pairs (correct > each incorrect answer)
            for i in range(4):  # There are 4 answer options (answer0, answer1, answer2, answer3)
                if i == label:
                    continue
                
                rejected = row[f"answer{i}"]
                prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer=chosen)
                pair = build_pair(prompt, chosen, rejected)
                f.write(json.dumps(pair) + "\n")

    print("Saved DPO dataset to", DATA_SAVE_PATH)

if __name__ == "__main__":
    main()
