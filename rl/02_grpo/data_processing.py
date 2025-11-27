import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

from datasets import load_dataset
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5-nano"
TOTAL_TARGET = 4000        # total samples
CONCURRENT_REQUESTS = 10   # number of parallel API calls
RPM_DELAY = 1.0            # delay (in seconds) between batches to respect RPM
OUTPUT_FILE = "../../data/processed/02_grpo/gsm8k_grpo_real.jsonl"

client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# PROMPT
# ==============================

WRONG_ANSWERS_PROMPT = """
You are creating negative training examples for a GSM8K-style dataset.

You are given:
1. A GSM8K math problem
2. The CORRECT answer in GSM8K format (with reasoning and final #### number)

Your job is to generate EXACTLY 3 WRONG answers that clearly follow the SAME GSM8K STYLE:

Formatting rules (IMPORTANT and REQUIRED):
- Provide step-by-step reasoning
- Include inline calculations using << >> like: <<2*4=8>>
- End with a final line in the format: #### <NUMBER>
- The final number MUST BE WRONG
- The explanation MUST be coherent and realistic

Types of mistakes you must generate:

A1: Small arithmetic mistake (logic correct, final number slightly wrong)
A2: Logical reasoning mistake (wrong method, wrong structure)
A3: Fluent but nonsensical explanation (still ends with #### <NUMBER>)

DO NOT repeat the correct answer.
DO NOT mention that it is wrong.
DO NOT explain the difference.

Return output in this exact format ONLY:

A1:
<answer here>

A2:
<answer here>

A3:
<answer here>

------------------------------------------------

Problem:
{question}

Correct Answer (for reference only):
{correct}
"""


def openai_call(prompt):
    """Issue a single request to OpenAI."""
    start_time = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        reasoning={
            "effort": "minimal"
        },
        text={ "verbosity": "low" }
    )
    elapsed_time = time.perf_counter() - start_time
    return response.output_text.strip(), elapsed_time


def process_sample(question, correct):
    """
    Generate wrong answers for a single GSM8K sample.

    Returns:
        dict with keys:
            record (dict or None)
            output (str)
            api_runtime (float)
            error (str or None)
    """
    prompt = WRONG_ANSWERS_PROMPT.format(
        question=question,
        correct=correct
    )
    output, api_runtime = openai_call(prompt)
    wrong_answers = parse_answers(output)

    if len(wrong_answers) != 3:
        return {
            "record": None,
            "output": output,
            "api_runtime": api_runtime,
            "error": "Bad format (expected 3 answers).",
        }

    record = {
        "prompt": question,
        "candidates": [correct] + wrong_answers,
        "ranks": [1, 2, 3, 4],
    }

    return {
        "record": record,
        "output": output,
        "api_runtime": api_runtime,
        "error": None,
    }


def batched(iterator, batch_size):
    """Yield lists of size batch_size from an iterator."""
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def parse_answers(text):
    """Extract A1, A2, A3 from OpenAI output"""
    answers = []
    for block in text.split("\n\n"):
        if block.strip().startswith("A"):
            lines = block.split("\n")
            cleaned = "\n".join(lines[1:]).strip()
            answers.append(cleaned)
    return answers


# ==============================
# LOAD GSM8K
# ==============================

print("\nLoading GSM8K dataset...")
dataset = load_dataset("openai/gsm8k", "main")["train"]
dataset = dataset.shuffle(seed=42).select(range(TOTAL_TARGET))

print(f"Loaded {len(dataset)} samples")

# ==============================
# MAIN LOOP
# ==============================

total_generated = 0
total_api_time = 0.0
sample_counter = 0

def sample_iterator():
    """Yield (seq_id, question, correct) tuples."""
    for idx, row in enumerate(dataset, start=1):
        yield idx, row["question"].strip(), row["answer"].strip()

with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor, open(OUTPUT_FILE, "w") as f:

    for batch in batched(sample_iterator(), CONCURRENT_REQUESTS):

        future_to_seq = {}
        for seq_id, question, correct in batch:
            if total_generated >= TOTAL_TARGET:
                break
            print(f"\n[{seq_id}/{TOTAL_TARGET}]  Using model: {MODEL}")
            future = executor.submit(process_sample, question, correct)
            future_to_seq[future] = (seq_id, question, correct)

        if not future_to_seq:
            break

        for future in as_completed(future_to_seq):
            seq_id, question, correct = future_to_seq[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"❌ Sample {seq_id}: {exc}")
                continue

            output = result["output"]
            api_runtime = result["api_runtime"]
            print(output)

            if result["record"] is None:
                print(f"⚠️ Sample {seq_id}: {result['error']}")
                continue

            f.write(json.dumps(result["record"]) + "\n")

            total_generated += 1
            total_api_time += api_runtime

            print(f"✅ Sample saved (Sample {seq_id}, API call: {api_runtime:.2f}s)")

        if total_generated >= TOTAL_TARGET:
            break

        time.sleep(RPM_DELAY)

# ==============================
# FINAL STATS
# ==============================

print("\n\n✅ DATA GENERATION COMPLETE")
print(f"\nTotal samples created: {total_generated}")
print(f"Total API time: {total_api_time:.2f}s")
if total_generated > 0:
    print(f"Average API call time: {total_api_time / total_generated:.2f}s")
print(f"Saved to: {OUTPUT_FILE}")
