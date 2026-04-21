import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from sacrebleu import corpus_bleu, corpus_chrf

# Allow running as: python bench/eval_gec.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_model, generate_corrections_with_scores
from src.suggestion_ranker import rank_suggestions


def load_pairs(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of lines in {path}")
    # Convention in this repo: [target(correct), input(incorrect)] repeating.
    return [(lines[i + 1], lines[i]) for i in range(0, len(lines), 2)]


@dataclass
class EvalResult:
    exact_match: float
    bleu: float
    chrf: float
    overcorrection_rate: float
    unchanged_rate_on_correct: float


@torch.inference_mode()
def evaluate(
    model_path: str,
    data_path: str,
    *,
    num_suggestions: int = 5,
    lambda_edit: float = 0.35,
    device: str | None = None,
) -> EvalResult:
    model, tokenizer = load_model(model_path)
    if device is not None:
        model.to(torch.device(device))

    pairs = load_pairs(data_path)
    preds: List[str] = []
    refs: List[str] = []

    correct_inputs = 0
    changed_on_correct = 0
    total_changed = 0

    for src, tgt in pairs:
        raw = generate_corrections_with_scores(model, tokenizer, src, num_suggestions=num_suggestions)
        ranked = rank_suggestions(
            original=src,
            suggestions=raw,
            metric="conservative",
            det_confidence=1.0,
            lambda_edit=lambda_edit,
        )
        pred = ranked[0][0] if ranked else src
        preds.append(pred)
        refs.append(tgt)

        if pred != src:
            total_changed += 1
        if src == tgt:
            correct_inputs += 1
            if pred != src:
                changed_on_correct += 1

    exact = sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs))
    bleu = corpus_bleu(preds, [refs]).score
    chrf = corpus_chrf(preds, [refs]).score

    overcorr_rate = changed_on_correct / max(1, correct_inputs)
    unchanged_rate_on_correct = 1.0 - overcorr_rate

    return EvalResult(
        exact_match=float(exact),
        bleu=float(bleu),
        chrf=float(chrf),
        overcorrection_rate=float(overcorr_rate),
        unchanged_rate_on_correct=float(unchanged_rate_on_correct),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local T5 fine-tuned model dir")
    ap.add_argument("--data", required=True, help="Path to paired data file (e.g. GEC/test.txt)")
    ap.add_argument("--k", type=int, default=5, help="Number of suggestions to generate")
    ap.add_argument("--lambda-edit", type=float, default=0.35, help="Edit penalty lambda")
    ap.add_argument("--device", default=None, help="cuda / cpu (optional)")
    args = ap.parse_args()

    res = evaluate(
        args.model,
        args.data,
        num_suggestions=args.k,
        lambda_edit=args.lambda_edit,
        device=args.device,
    )
    print("exact_match:", res.exact_match)
    print("BLEU:", res.bleu)
    print("chrF:", res.chrf)
    print("overcorrection_rate:", res.overcorrection_rate)
    print("unchanged_rate_on_correct:", res.unchanged_rate_on_correct)


if __name__ == "__main__":
    main()

