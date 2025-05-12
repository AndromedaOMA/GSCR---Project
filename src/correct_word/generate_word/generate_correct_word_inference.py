from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_dir = "ro_inflection_corrector-model"
model = MT5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = MT5Tokenizer.from_pretrained(model_dir)

def suggest_corrections(word, top_k=3):
    input_ids = tokenizer.encode(word, return_tensors="pt")
    outputs = model.generate(input_ids, num_beams=5, num_return_sequences=top_k,
                              early_stopping=True)
    suggestions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
    return suggestions

word = "merge"
candidates = suggest_corrections(word, top_k=3)
print(candidates)
