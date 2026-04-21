import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path: str):

    # Loads tokenizer + model.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    # if you have a GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

@torch.inference_mode()
def _sequence_avg_logprob_from_generate(gen_out, sequences, tokenizer):
    """
    Approximate per-sequence average log-probability from `generate` scores.

    Notes:
    - Works with `return_dict_in_generate=True, output_scores=True`.
    - Uses step-wise scores aligned to generated tokens after the prompt.
    - For beam search, this is a good practical proxy for P(s|model) used in ranking.
    """
    if not hasattr(gen_out, "scores") or gen_out.scores is None:
        return [None for _ in range(sequences.shape[0])]

    scores = gen_out.scores  # list[len=T] of (batch*num_return, vocab)
    # Generated token ids typically include a leading decoder_start/eos and then tokens.
    # Align by taking the last T tokens of each sequence (T == len(scores)).
    T = len(scores)
    seq_tail = sequences[:, -T:]
    avg_logps = []
    for i in range(sequences.shape[0]):
        logp_sum = 0.0
        tok_count = 0
        for t in range(T):
            step_scores = scores[t][i]  # (vocab,)
            step_logp = torch.log_softmax(step_scores, dim=-1)[seq_tail[i, t]].item()
            logp_sum += step_logp
            tok_count += 1
        avg_logps.append(logp_sum / max(1, tok_count))
    return avg_logps

def generate_corrections(model, tokenizer, text: str, num_suggestions: int = 3):
    
    # Generate the top n suggestions for the given text.
    device = next(model.parameters()).device
    inputs = tokenizer.encode("grammar: " + text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        num_beams=num_suggestions,
        num_return_sequences=num_suggestions,
        max_length=inputs.shape[-1] + 50,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    suggestions = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

def generate_corrections_with_scores(model, tokenizer, text: str, num_suggestions: int = 3):
    """
    Like `generate_corrections`, but also returns an approximate model score per suggestion.

    Returns:
      List[dict]: [{"text": str, "avg_logprob": float}, ...]
    """
    device = next(model.parameters()).device
    inputs = tokenizer.encode("grammar: " + text, return_tensors="pt").to(device)
    gen_out = model.generate(
        inputs,
        num_beams=num_suggestions,
        num_return_sequences=num_suggestions,
        max_length=inputs.shape[-1] + 50,
        early_stopping=True,
        no_repeat_ngram_size=3,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = gen_out.sequences
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in sequences]
    avg_logps = _sequence_avg_logprob_from_generate(gen_out, sequences, tokenizer)

    seen = set()
    out = []
    for txt, lp in zip(decoded, avg_logps):
        if txt in seen:
            continue
        seen.add(txt)
        out.append({"text": txt, "avg_logprob": float(lp) if lp is not None else None})
    return out
