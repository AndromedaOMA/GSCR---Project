import sqlite3
from pathlib import Path

from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

def run_active_learning(
    model,
    tokenizer,
    db_path: str,
    output_dir: str = "../../t5-grammar-finetuned",
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
):
    """
    1) Reads all rows from the `feedback` table in the sqlite DB.
    2) Builds a tiny fine-tuning dataset: inputs = "grammar: " + original, targets = chosen suggestion.
    3) Runs one epoch of further fine-tuning on that data.
    4) Saves model/tokenizer back to `output_dir`.
    """
    # load feedback
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT original, chosen FROM feedback")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("[active_learning] no feedback yet, skipping.")
        return

    originals, choosens = zip(*rows)

    # build HF Dataset
    ds = Dataset.from_dict({
        "input": [f"grammar: {t}" for t in originals],
        "target": list(choosens)
    })

    def tokenize_fn(batch):
        return tokenizer(
            batch["input"],
            text_target=batch["target"],
            max_length=96,
            max_target_length=128,
            padding="max_length",
            truncation=True,
        )
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    # set up Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="no",               # no intermediate checkpoints
        logging_dir=f"{output_dir}/al-logs",
        logging_strategy="epoch",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"[active_learning] fine-tuning on {len(tokenized)} feedback examples...")
    trainer.train()

    # persist
    Path(output_dir).mkdir(exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[active_learning] done; model saved to {output_dir}.")
