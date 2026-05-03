#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Romanian grammatical-correctness detector inference utilities.

Checkpoint loading avoids redundant Hub pulls (see preprocess.detector_inference_loader).
"""

import os
import re
from typing import Tuple

import torch
from transformers import AutoTokenizer, PreTrainedModel

from src.preprocess.detector_inference_loader import load_detector_for_inference
from src.preprocess.HFWrapperULMFiT import HFWrapperULMFiT

__all__ = [
    "HFWrapperULMFiT",
    "load_detector_for_inference",
    "clean_text",
    "predict_on_text",
]


def clean_text(text: str) -> str:
    """Remove leading/trailing spaces and collapse excessive whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def load_detector(
    model_path: str, device: torch.device | None = None
) -> Tuple[HFWrapperULMFiT, AutoTokenizer]:
    """Load tokenizer + classifier from a local Trainer export directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_detector_for_inference(model_path, device)


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
    label_map = {0: "Correct", 1: "Incorrect"}
    return label_map[pred_idx], pred_idx


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, "content", "trained_model_V2_2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_detector_for_inference(local_model_path, device)

    examples = [
        "El merge la școală în fiecare zi.",
        "Mam mai gândit că poate voia să țepuiască pe cineva.",
        "Ea este femeia pe care am văzut-o.",
        "La sfarsit de an a mers acolo.",
    ]

    for s in examples:
        verdict, _ = predict_on_text(s, model, tokenizer)
        print(f"“{s}” -> {verdict}")


if __name__ == "__main__":
    main()
