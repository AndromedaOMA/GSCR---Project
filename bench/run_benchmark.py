import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sacrebleu import corpus_bleu, corpus_chrf

# Allow running as: python bench/run_benchmark.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import generate_corrections, generate_corrections_with_scores, load_model
from src.preprocess.detector_inference_loader import load_detector_for_inference
from src.suggestion_ranker import rank_suggestions


def load_pairs(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of non-empty lines in {path}")
    # Repo convention: target(correct), input(incorrect), repeating.
    return [(lines[i + 1], lines[i]) for i in range(0, len(lines), 2)]


@dataclass
class Metrics:
    exact_match: float
    bleu: float
    chrf: float
    overcorrection_rate: float
    unchanged_rate_on_correct: float
    changed_rate: float


class DetectorConfidence:
    def __init__(self, detector_path: str, device: torch.device):
        local_path = Path(detector_path)
        candidates = []
        if local_path.is_absolute():
            candidates.append(local_path)
        else:
            candidates.append(PROJECT_ROOT / local_path)
            # Common alternative users provide: path relative to repo root or script dir.
            candidates.append((Path.cwd() / local_path).resolve())
            # Mirror the path hardcoded in app.py for easier reuse.
            candidates.append(PROJECT_ROOT / "src" / "detection" / "content" / "trained_model_V2_2")

        existing = next((p.resolve() for p in candidates if p.exists()), None)
        if existing is None:
            tried = "\n".join(f"  - {str(p)}" for p in candidates)
            raise FileNotFoundError(
                "Detector checkpoint path was not found locally. Tried:\n"
                f"{tried}\n"
                "Pass a valid local folder that contains tokenizer/model files."
            )

        model_ref = str(existing)
        self.model, self.tokenizer = load_detector_for_inference(model_ref, device)
        self.device = device

    @torch.inference_mode()
    def prob_incorrect(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        logits = self.model(**enc)["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[1].item())


def compute_metrics(preds: Sequence[str], refs: Sequence[str], srcs: Sequence[str]) -> Metrics:
    exact = sum(p == r for p, r in zip(preds, refs)) / max(1, len(refs))
    bleu = corpus_bleu(list(preds), [list(refs)]).score
    chrf = corpus_chrf(list(preds), [list(refs)]).score

    total = len(srcs)
    changed = sum(p != s for p, s in zip(preds, srcs))
    correct_inputs = sum(s == r for s, r in zip(srcs, refs))
    changed_on_correct = sum((p != s) and (s == r) for p, s, r in zip(preds, srcs, refs))
    overcorr = changed_on_correct / max(1, correct_inputs)

    return Metrics(
        exact_match=float(exact),
        bleu=float(bleu),
        chrf=float(chrf),
        overcorrection_rate=float(overcorr),
        unchanged_rate_on_correct=float(1.0 - overcorr),
        changed_rate=float(changed / max(1, total)),
    )


def predict_strategy(
    strategy: str,
    src: str,
    model,
    tokenizer,
    num_suggestions: int,
    lambda_edit: float,
    det_confidence: float,
) -> str:
    if strategy == "t5_top1":
        raw = generate_corrections(model, tokenizer, src, num_suggestions=1)
        return raw[0] if raw else src

    if strategy == "bleu_ranker":
        raw = generate_corrections(model, tokenizer, src, num_suggestions=num_suggestions)
        ranked = rank_suggestions(original=src, suggestions=raw, metric="bleu")
        return ranked[0][0] if ranked else src

    raw_scored = generate_corrections_with_scores(model, tokenizer, src, num_suggestions=num_suggestions)

    if strategy == "conservative":
        ranked = rank_suggestions(
            original=src,
            suggestions=raw_scored,
            metric="conservative",
            det_confidence=det_confidence,
            lambda_edit=lambda_edit,
        )
        return ranked[0][0] if ranked else src

    if strategy == "conservative_no_detconf":
        ranked = rank_suggestions(
            original=src,
            suggestions=raw_scored,
            metric="conservative",
            det_confidence=1.0,
            lambda_edit=lambda_edit,
        )
        return ranked[0][0] if ranked else src

    if strategy == "conservative_no_edit":
        ranked = rank_suggestions(
            original=src,
            suggestions=raw_scored,
            metric="conservative",
            det_confidence=det_confidence,
            lambda_edit=0.0,
        )
        return ranked[0][0] if ranked else src

    raise ValueError(f"Unknown strategy: {strategy}")


@torch.inference_mode()
def evaluate_dataset(
    data_path: str,
    strategies: Sequence[str],
    model,
    tokenizer,
    num_suggestions: int,
    lambda_edit: float,
    detector: Optional[DetectorConfidence],
) -> Dict[str, Metrics]:
    pairs = load_pairs(data_path)
    srcs = [src for src, _ in pairs]
    refs = [tgt for _, tgt in pairs]

    preds_by_strategy: Dict[str, List[str]] = {s: [] for s in strategies}

    for src in srcs:
        det_conf = detector.prob_incorrect(src) if detector is not None else 1.0
        for strategy in strategies:
            pred = predict_strategy(
                strategy=strategy,
                src=src,
                model=model,
                tokenizer=tokenizer,
                num_suggestions=num_suggestions,
                lambda_edit=lambda_edit,
                det_confidence=det_conf,
            )
            preds_by_strategy[strategy].append(pred)

    out: Dict[str, Metrics] = {}
    for strategy in strategies:
        out[strategy] = compute_metrics(preds_by_strategy[strategy], refs, srcs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local T5 fine-tuned model directory")
    ap.add_argument("--out", default="bench/results.json", help="Output JSON path")
    ap.add_argument("--k", type=int, default=5, help="Number of T5 hypotheses")
    ap.add_argument("--lambda-edit", type=float, default=0.35, help="Edit penalty lambda")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--detector-path", default=None, help="Optional detector path (HFWrapperULMFiT checkpoint)")
    ap.add_argument("--level1", default="GEC/test.txt", help="Level 1 dataset path")
    ap.add_argument("--level2", default="GEC/dev.txt", help="Level 2 dataset path")
    ap.add_argument(
        "--level3",
        default="bench/level3_stress_from_gec.txt",
        help="Level 3 dataset path",
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "t5_top1",
            "bleu_ranker",
            "conservative",
            "conservative_no_detconf",
            "conservative_no_edit",
        ],
    )
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_model(args.model)
    model.to(device)

    detector = None
    if args.detector_path:
        try:
            detector = DetectorConfidence(args.detector_path, device=device)
            print("[INFO] Detector loaded; reranking will use per-sentence detector confidence.")
        except Exception as exc:
            print(f"[WARN] Detector disabled: {exc}")
            print("[WARN] Continuing with det_confidence=1.0 for all samples.")

    levels = {
        "level1_in_distribution": args.level1,
        "level2_stress": args.level2,
        "level3_authentic": args.level3,
    }

    detector_enabled = detector is not None
    report = {
        "config": {
            "model": args.model,
            "k": args.k,
            "lambda_edit": args.lambda_edit,
            "device": str(device),
            "detector_path": args.detector_path,
            "detector_enabled": detector_enabled,
            "strategies": args.strategies,
        },
        "results": {},
    }

    for level_name, path in levels.items():
        if not Path(path).exists():
            report["results"][level_name] = {"status": "missing_dataset", "path": path}
            continue
        metrics = evaluate_dataset(
            data_path=path,
            strategies=args.strategies,
            model=model,
            tokenizer=tokenizer,
            num_suggestions=args.k,
            lambda_edit=args.lambda_edit,
            detector=detector,
        )
        report["results"][level_name] = {
            "path": path,
            "num_examples": len(load_pairs(path)),
            "metrics": {k: asdict(v) for k, v in metrics.items()},
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote benchmark report to {out_path}")


if __name__ == "__main__":
    main()

