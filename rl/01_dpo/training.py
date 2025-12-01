import logging

import torch
from constants import (ALPHA, BATCH_SIZE, BETA, DATA_SAVE_PATH, DEVICE,
                       LEARNING_RATE, LOG_FILE, LOG_FORMAT, LOGGING_INTERVAL,
                       MAX_SEQ_LEN, MODEL_NAME, NUM_EPOCHS, NUM_SAMPLES, SEED,
                       SPLIT_RATIO, R)
from custom_dataset import CosmosQADataset
from dpo_utils import dpo_loss, kl_divergence, sequence_logprobs
from lora import inject_lora
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

############################################################
# LOAD MODEL + TOKENIZER
############################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
ref_model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

policy_model = inject_lora(policy_model, R, ALPHA)
policy_model.train()


############################################################
# LOAD DATASET
############################################################

full_dataset = CosmosQADataset(DATA_SAVE_PATH, tokenizer, MAX_SEQ_LEN, NUM_SAMPLES)

# 90% train, 10% test split
train_size = int(SPLIT_RATIO * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


############################################################
# ACCURACY FUNCTION
############################################################

def evaluate_accuracy(model, ref_model, loader):
    """
    Accuracy = % of pairs where:
       policy_logprob(chosen) > policy_logprob(rejected)
    """
    correct = 0
    total   = 0

    model.eval()
    ref_model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating accuracy"):

            chosen_ids = batch["chosen_ids"].to(DEVICE)
            chosen_mask = batch["chosen_mask"].to(DEVICE)
            rejected_ids = batch["rejected_ids"].to(DEVICE)
            rejected_mask = batch["rejected_mask"].to(DEVICE)

            p_chosen = sequence_logprobs(model, chosen_ids, chosen_mask)
            p_reject = sequence_logprobs(model, rejected_ids, rejected_mask)

            # A correct preference is p(chosen) > p(rejected)
            correct += (p_chosen > p_reject).sum().item()
            total   += chosen_ids.size(0)

    model.train()
    return correct / total


############################################################
# PRE-TRAINING EVALUATION
############################################################

logger.info("Evaluating BEFORE training...")
acc_before = evaluate_accuracy(policy_model, ref_model, test_loader)
logger.info("Accuracy before training: %.4f", acc_before)


############################################################
# OPTIMIZER
############################################################

optimizer = torch.optim.AdamW(
    [p for p in policy_model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)


############################################################
# TRAINING LOOP
############################################################

global_step = 0

for epoch in range(NUM_EPOCHS):
    logger.info("====== Epoch %d/%d ======", epoch + 1, NUM_EPOCHS)

    for batch in tqdm(train_loader):

        chosen_ids = batch["chosen_ids"].to(DEVICE)
        chosen_mask = batch["chosen_mask"].to(DEVICE)
        rejected_ids = batch["rejected_ids"].to(DEVICE)
        rejected_mask = batch["rejected_mask"].to(DEVICE)

        #############################
        # 1️⃣ Policy (LoRA model) forward
        #############################
        # log-prob scores for the DPO objective
        policy_chosen = sequence_logprobs(policy_model, chosen_ids,   chosen_mask)
        policy_reject = sequence_logprobs(policy_model, rejected_ids, rejected_mask)

        #############################
        # 2️⃣ Reference (frozen) forward
        #############################
        with torch.no_grad():
            ref_chosen = sequence_logprobs(ref_model, chosen_ids,   chosen_mask)
            ref_reject = sequence_logprobs(ref_model, rejected_ids, rejected_mask)

        #############################
        # 3️⃣ DPO Loss
        #############################
        loss = dpo_loss(policy_chosen, policy_reject, ref_chosen, ref_reject, BETA)

        #############################
        # 4️⃣ Backward + Step
        #############################
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1

        #############################
        # 5️⃣ Logging
        #############################
        if global_step % LOGGING_INTERVAL == 0:
            # KL Divergence (only computed during logging to reduce overhead)
            # Forward passes for logits (only when logging) - no_grad to avoid memory accumulation
            with torch.no_grad():
                policy_out_ch = policy_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                policy_out_rj = policy_model(input_ids=rejected_ids, attention_mask=rejected_mask)
                ref_out_ch = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask)
                ref_out_rj = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask)
            
            policy_logits_ch = policy_out_ch.logits[:, :-1, :]
            policy_logits_rj = policy_out_rj.logits[:, :-1, :]
            ref_logits_ch = ref_out_ch.logits[:, :-1, :]
            ref_logits_rj = ref_out_rj.logits[:, :-1, :]
            
            labels_ch = chosen_ids[:, 1:]
            labels_rj = rejected_ids[:, 1:]

            kl_ch = kl_divergence(policy_logits_ch, ref_logits_ch, labels_ch)
            kl_rj = kl_divergence(policy_logits_rj, ref_logits_rj, labels_rj)

            kl_avg = 0.5 * (kl_ch + kl_rj)
            log_message = LOG_FORMAT.format(step=global_step, loss=loss.item(), kl=kl_avg.item())
            logger.info(log_message)

    acc_after = evaluate_accuracy(policy_model, ref_model, test_loader)
    logger.info("Accuracy after %d epochs: %.4f", epoch + 1, acc_after)

logger.info("Accuracy improvement:")
logger.info("Δ accuracy = %.4f", acc_after - acc_before)

