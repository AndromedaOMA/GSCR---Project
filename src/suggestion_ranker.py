# src/suggestion_ranker.py

from sacrebleu import sentence_bleu

def rank_suggestions(original, suggestions, metric="bleu"):
    """
    Given an original sentence and a list of candidate corrections,
    return a list of (suggestion, score) sorted by score descending.
    Uses sacrebleu.sentence_bleu under the hood.
    """
    scored = []
    for cand in suggestions:
        if metric == "bleu":
            bleu = sentence_bleu([cand], [[original]])
            score = bleu.score
        else:
            score = 0.0

        scored.append((cand, score))

    # sort highest→lowest
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
