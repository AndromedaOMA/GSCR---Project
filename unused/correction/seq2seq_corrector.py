from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_corrections(text, num_options=5):
    # Generates multiple correction options for a given text
    tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")
    model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")

    prefix = "grammar: "
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**inputs, num_return_sequences=num_options, num_beams=num_options)

    options = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return options