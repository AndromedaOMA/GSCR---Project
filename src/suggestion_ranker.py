from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from sacrebleu import sentence_bleu

from src.correct_word.levenshtein import levenshtein

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _to_text_and_score_list(
    suggestions: Sequence[Union[str, dict]]
) -> List[Tuple[str, Optional[float]]]:
    out: List[Tuple[str, Optional[float]]] = []
    for s in suggestions:
        if isinstance(s, str):
            out.append((s, None))
        elif isinstance(s, dict):
            out.append((str(s.get("text", "")), s.get("avg_logprob", None)))
        else:
            out.append((str(s), None))
    return out

def rank_suggestions(
    original: str,
    suggestions: Sequence[Union[str, dict]],
    metric: str = "bleu",
    *,
    det_confidence: float = 1.0,
    lambda_edit: float = 0.35,
) -> List[Tuple[str, float]]:
    scored = []
    det_conf = max(0.0, min(1.0, _safe_float(det_confidence, 1.0)))
    pairs = _to_text_and_score_list(suggestions)
    for cand, avg_logprob in pairs:
        cand = cand.strip()
        if not cand:
            continue

        if metric == "bleu":
            # Backward-compatible baseline: prefer minimal change (BLEU vs original).
            bleu = sentence_bleu(cand, [original])
            score = float(bleu.score)
        elif metric in {"conservative", "reranker", "uncertainty"}:
            # Proposed CIKM paper ranking:
            #   Score(s) = det_conf * log P_T5(s | x) - lambda * Lev(original, s)
            # Using logP (avg_logprob) is numerically stable and preserves ordering.
            lp = _safe_float(avg_logprob, default=-50.0)
            edit = levenshtein(original, cand)
            score = det_conf * lp - float(lambda_edit) * float(edit)
        else:
            score = 0.0

        scored.append((cand, float(score)))

    # sort highest -> lowest
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored