from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Trainer, TrainingArguments

# Load a pretrained Romanian-compatible seq2seq model and tokenizer
model_name = "google/mt5-small"  # multilingual T5-small (300M)
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Prepare dataset
train_pairs = [{"input": inp, "target": tgt} for inp, tgt in zip(train_inputs, train_targets)]
dev_pairs = [{"input": inp, "target": tgt} for inp, tgt in zip(dev_inputs, dev_targets)]

# Tokenize the dataset
def tokenize_batch(batch):
    model_inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=16)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=16)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

from datasets import Dataset
train_data = Dataset.from_list(train_pairs).map(tokenize_batch, batched=True, remove_columns=["input", "target"])
dev_data = Dataset.from_list(dev_pairs).map(tokenize_batch, batched=True, remove_columns=["input", "target"])

# Set training parameters
training_args = TrainingArguments(
    output_dir="ro_inflect_corrector",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=500
)

# Initialize Trainer
trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_data, eval_dataset=dev_data)

# Train the model
trainer.train()
trainer.save_model("ro_inflection_corrector-model")
