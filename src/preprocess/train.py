import sys
import pathlib
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer, AutoConfig

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.preprocess.RoNACCDatasetPaired import RoNACCDatasetPaired
from src.preprocess.HFWrapperULMFiT import HFWrapperULMFiT
from src.preprocess.utils import compute_metrics


print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

base_path = "../GEC"
train_path = f"{base_path}/train.txt"
val_path = f"{base_path}/dev.txt"
test_path = f"{base_path}/test.txt"

# Load datasets
train_dataset = RoNACCDatasetPaired(train_path, tokenizer)
val_dataset = RoNACCDatasetPaired(val_path, tokenizer)
test_dataset = RoNACCDatasetPaired(test_path, tokenizer)

# Debug statistics
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
for batch in train_loader:
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Labels:", batch["labels"])
    break

config = AutoConfig.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", num_labels=2)
model = HFWrapperULMFiT(config)


# TRAIN
training_args = TrainingArguments(
    output_dir="./ronacc_model",
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()


# INFERENCE
predictions = trainer.predict(test_dataset)
print("Evaluation metrics on test set:", predictions.metrics)

# Decode input texts for CSV export
decoded_texts = [
    tokenizer.decode(test_dataset[i]["input_ids"], skip_special_tokens=True)
    for i in range(len(test_dataset))
]

# Convert predictions
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# Save results
df = pd.DataFrame({
    "text": decoded_texts,
    "true_label": true_labels,
    "predicted_label": predicted_labels
})

df.to_csv("ronacc_test_predictions.csv", index=False)
print("Saved test predictions to ronacc_test_predictions.csv")