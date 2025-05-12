#!/usr/bin/env python3
# src/correct_word/generate_word/generate_correct_word_train.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

# ─────────────── CONFIG ───────────────────────────────────────────────────────
set_seed(42)

MODEL_NAME                  = "BlackKakapo/t5-small-grammar-ro-root"
DATA_CSV                    = os.path.join("GEC", "word_correction_pairs_augmented.csv")
OUTPUT_DIR                  = "ro-word-correction"
MAX_INPUT_LENGTH            = 16
MAX_TARGET_LENGTH           = 16
NUM_TRAIN_EPOCHS            = 5
PER_DEVICE_BATCH            = 32
LEARNING_RATE               = 5e-5

# how often to run evaluation & checkpointing (in steps)
EVAL_STEPS                  = 500
SAVE_STEPS                  = 500
LOGGING_STEPS               = 100
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 2) Read your augmented CSV
    df = pd.read_csv(DATA_CSV, encoding="utf-8")
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)

    # 3) Convert to HF Dataset
    train_ds = Dataset.from_pandas(train_df)
    dev_ds   = Dataset.from_pandas(dev_df)

    # 4) Tokenization / preprocessing
    def preprocess_fn(batch):
        enc = tokenizer(
            batch["wrong"],
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
        )
        tgt = tokenizer(
            text_target=batch["correct"],
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )
        enc["labels"] = tgt["input_ids"]
        return enc

    train_ds = train_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_ds.column_names
    )
    dev_ds = dev_ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=dev_ds.column_names
    )

    # 5) Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 6) Exact-match metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # replace -100 with pad_token_id for decoding:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        matches = [p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels)]
        return {"exact_match": sum(matches) / len(matches)}

    # 7) Training arguments (old‐style: eval_steps & save_steps, no evaluation_strategy)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,

        # old‐style eval / save
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=2,

        # generate‐at‐eval time
        predict_with_generate=True,
    )

    # 8) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9) Train & save
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Word‐level model fine-tuned and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
