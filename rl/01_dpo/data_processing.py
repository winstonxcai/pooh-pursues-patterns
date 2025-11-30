import json

from constants import DATA_SAVE_PATH, DATASET_NAME
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
    data = load_dataset(DATASET_NAME)["train"]

    with open(DATA_SAVE_PATH, "w") as f:
        for row in tqdm(data, desc="Processing dataset"):

            context = row["context"]
            question = row["question"]
            options = row["answers"]
            label = row["label"]

            prompt = f"Context: {context}\nQuestion: {question}\nAssistant:"
            chosen = options[label]

            # generate 3 pairs (correct > each incorrect)
            for i, opt in enumerate(options):
                if i == label:
                    continue
                
                rejected = opt
                pair = build_pair(prompt, chosen, rejected)
                f.write(json.dumps(pair) + "\n")

    print("Saved DPO dataset to", DATA_SAVE_PATH)

if __name__ == "__main__":
    main()
