#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Romanian grammatical-correctness detector and export HF checkpoint.

This script trains HFWrapperULMFiT on paired GEC data and saves:
- model weights/config via save_pretrained()
- tokenizer via save_pretrained()

Default output is compatible with app.py and bench/run_benchmark.py:
    src/detection/content/trained_model_V2_2
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.HFWrapperULMFiT import HFWrapperULMFiT
from src.preprocess.RoNACCDatasetPaired import RoNACCDatasetPaired
from src.preprocess.utils import compute_metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="GEC/train.txt")
    ap.add_argument("--dev", default="GEC/dev.txt")
    ap.add_argument("--test", default="GEC/test.txt")
    ap.add_argument("--base-model", default="dumitrescustefan/bert-base-romanian-cased-v1")
    ap.add_argument("--out", default="src/detection/content/trained_model_V2_2")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def resolve_path(p: str) -> pathlib.Path:
    path = pathlib.Path(p)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = resolve_path(args.train)
    dev_path = resolve_path(args.dev)
    test_path = resolve_path(args.test)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists() or not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing data files:\n"
            f"  train={train_path}\n"
            f"  dev={dev_path}\n"
            f"  test={test_path}"
        )

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_dataset = RoNACCDatasetPaired(str(train_path), tokenizer, max_length=args.max_length)
    dev_dataset = RoNACCDatasetPaired(str(dev_path), tokenizer, max_length=args.max_length)
    test_dataset = RoNACCDatasetPaired(str(test_path), tokenizer, max_length=args.max_length)

    config = AutoConfig.from_pretrained(args.base_model, num_labels=2)
    model = HFWrapperULMFiT(config)

    run_dir = out_dir / "trainer_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=max(10, args.eval_steps // 5),
        save_steps=args.save_steps,
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    test_metrics = trainer.predict(test_dataset).metrics
    print("Test metrics:", test_metrics)

    # Export exactly what inference expects.
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics_path = out_dir / "metrics_test.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved detector checkpoint to: {out_dir}")
    print(f"Saved test metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

