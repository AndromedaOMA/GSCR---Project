from sacrebleu import sentence_bleu

def rank_suggestions(original, suggestions, metric="bleu"):
    scored = []
    for cand in suggestions:
        if metric == "bleu":
            # pass cand (str) as the hypothesis, and [original] as the list of refs
            bleu = sentence_bleu(cand, [original])
            score = bleu.score
        else:
            score = 0.0

        scored.append((cand, score))

    # sort highest -> lowest
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored