import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset
from typing import List, Dict
from sacrebleu import corpus_bleu
from transformers import EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq
from transformers import set_seed
set_seed(42)


# Config
model_name = "BlackKakapo/t5-small-grammar-ro-root"
base_path = "../../GEC"
train_path = f"{base_path}/train.txt"
eval_path = f"{base_path}/dev.txt"
test_path = f"{base_path}/test.txt"
output_dir = "../../t5-grammar-finetuned"
max_input_length = 96
max_target_length = 128
num_train_epochs = 10
batch_size = 16
learning_rate = 5e-5
eval_steps = 100

# Initializing tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load data correct-incorrect pairs
def load_sentence_pairs(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) % 2 == 0, f"Fișierul '{path}' trebuie să aibă un număr par de linii."
    return [{"input": lines[i+1], "target": lines[i]} for i in range(0, len(lines), 2)]

def preprocess(example):
    return tokenizer(
        "grammar: " + example["input"],
        text_target=example["target"],
        max_length=max_input_length,   # applies to both
        padding="max_length",
        truncation=True,
    )


# Tokenizing and preprocessing
train_dataset = Dataset.from_list(load_sentence_pairs(train_path)).map(preprocess)
eval_dataset = Dataset.from_list(load_sentence_pairs(eval_path)).map(preprocess)
test_dataset = Dataset.from_list(load_sentence_pairs(test_path)).map(preprocess)


# Metric only for generational AI
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # convert to lists and clamp out-of-range tokens exactly as before…
    preds = preds.tolist() if hasattr(preds, "tolist") else preds
    labels = labels.tolist() if hasattr(labels, "tolist") else labels
    max_id = tokenizer.vocab_size - 1
    clamped = [[min(max_id, max(0, t)) for t in seq] for seq in preds]
    decoded_preds  = tokenizer.batch_decode(clamped, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # now calculate BLEU instead of exact-match
    bleu = corpus_bleu(decoded_preds, [decoded_labels])
    return {
        "bleu": bleu.score,                   # BLEU score 0–100
        "exact_match": sum(p==l for p,l in zip(decoded_preds,decoded_labels)) / len(decoded_labels)
    }


# Config training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size, # batch size for training
    per_device_eval_batch_size=batch_size, # batch size for eval
    num_train_epochs=num_train_epochs, # number of epochs for training
    learning_rate=learning_rate, # learning rate - small, we are finetuning a model(tweaking the model)
    weight_decay=0.01,
    eval_strategy="epoch", # eval for epochs, not on steps
    eval_steps=eval_steps,
    save_strategy="epoch", # saving on epochs
    logging_strategy="steps", # logging each step
    logging_steps=10,
    predict_with_generate=True,
    logging_dir=os.path.join(output_dir, "logs"),
    save_total_limit=2,
    report_to="none", # can activate "Weights & Biases" if you have an account
    load_best_model_at_end=True,
    metric_for_best_model="bleu", # criterion of rbest model is the loss on evaluation, small loss -> good model
    greater_is_better=True,
    # fp16=False, # 2 commented args if you don't have GPU and training on CPU or TPU
    # bf16=True,
    fp16=True, # GPU ONLYYYYYY !!!!!!!!!!!!!!!!!!!!!!!!
    fp16_full_eval=False, # GPU ONLYYYYYY !!!!!!!!!!!!!!!!!!!!!!!!

)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")


# Initializing trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Antrenament finalizat. Model salvat în:", output_dir)

# Eval model
results = trainer.evaluate(test_dataset)
print("🧪 Rezultate evaluare test:", results)

# Module with caching and ranking of sugestions
class RomanianGrammarCorrector:
    def __init__(self, model_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.suggestion_cache = {}
        self.common_mistakes = [
            "sa fi sigur ca",
            "ia-ti cartea",
            "el vroia sa mearga"
        ]
        for mistake in self.common_mistakes:
            self.suggest_corrections(mistake, use_cache=False)

    def suggest_corrections(self, text: str, max_suggestions: int = 5, use_cache: bool = True):
        if use_cache and text in self.suggestion_cache:
            return [s[0] for s in self.suggestion_cache[text]]
        input_text = "grammar: " + text
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs,
            num_beams=max_suggestions,
            num_return_sequences=max_suggestions,
            max_length=len(inputs[0]) + 50,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        suggestions = list(dict.fromkeys(self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs))
        suggestion_scores = [(s, None) for s in suggestions]
        if use_cache:
            self.suggestion_cache[text] = suggestion_scores
        return suggestions

    def rank_suggestions(self, suggestions: List[str], original_text: str = None):
        return suggestions

if __name__ == "__main__":
    corrector = RomanianGrammarCorrector(model_path=output_dir)
    text_gresit = "Eu si prietenul meu au mers ieri in parc, dar nu am cumparat nimic inafara de o carte."
    sugestii = corrector.suggest_corrections(text_gresit, max_suggestions=3)
    sugestii_rankate = corrector.rank_suggestions(sugestii, original_text=text_gresit)
    print("Sugestii corectare pentru:", text_gresit)
    for idx, sug in enumerate(sugestii_rankate, 1):
        print(f"{idx}. {sug}")
