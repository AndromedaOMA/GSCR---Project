#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Romanian grammatical-correctness detector
Loads the fine-tuned HFWrapperULMFiT model saved with
`model.save_pretrained(<output_dir>)` and performs inference.
"""

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedModel,
    BertConfig,
)

# ---------------------------------------------------------------------
# 1.  Classifier head used during training (ULMFiT-style pooling)
# ---------------------------------------------------------------------
class ULMFiTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "dumitrescustefan/bert-base-romanian-cased-v1",
        dropout: float = 0.15,
        num_labels: int = 2,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Concatenate CLS, mean-pool and max-pool representations
        self.pooler = lambda x: torch.cat(
            (x[:, 0], torch.mean(x, dim=1), torch.max(x, dim=1).values), dim=1
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled = self.pooler(sequence_output)
        logits = self.fc(pooled)
        return logits


# ---------------------------------------------------------------------
# 2.  Wrapper so the model can be saved / loaded with Hugging Face
# ---------------------------------------------------------------------
class HFWrapperULMFiT(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__(config)
        # config._name_or_path points at the original base model
        self.ulmfit_model = ULMFiTClassifier(model_name=config._name_or_path)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # **kwargs will catch token_type_ids, position_ids, etc.
        logits = self.ulmfit_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------
# 3.  Helpers
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove leading/trailing spaces and collapse excessive whitespace."""
    return re.sub(r"\s+", " ", text.strip())


@torch.inference_mode()
def predict_on_text(
    text: str,
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
):
    text = clean_text(text)
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)

    logits = model(**encoding)["logits"]
    pred_idx = torch.argmax(logits, dim=1).item()
    label_map = {0: "✅ Correct", 1: "❌ Incorrect"}
    return label_map[pred_idx], pred_idx

# ---------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------
def main():
    # Build a **relative** path so HF never mistakes it for a Hub repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, "content", "trained_model_V2_2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import your custom class before loading the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = HFWrapperULMFiT.from_pretrained(local_model_path).to(device)
    model.eval()

    # Demo sentences
    examples = [
        "El merge la școală în fiecare zi.",
        "Mam mai gândit că poate voia să țepuiască pe cineva.",
        "Ea este femeia pe care am văzut-o.",
        "La sfarsit de an a mers acolo."
    ]

    for s in examples:
        verdict, _ = predict_on_text(s, model, tokenizer)
        print(f"“{s}” → {verdict}")


if __name__ == "__main__":
    main()