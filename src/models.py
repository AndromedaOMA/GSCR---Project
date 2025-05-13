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
