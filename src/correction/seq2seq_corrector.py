from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Încarcă tokenizer-ul și modelul
tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")
model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")

if __name__ == "__main__":
    # Textul de corectat
    sent = "Miar placea sa merg la munca"
    prefix = "grammar: "
    example = prefix + sent

    # Tokenizare
    inputs = tokenizer(example, return_tensors="pt")

    # Generare corectare
    outputs = model.generate(**inputs)

    # Decodificare rezultat
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Text corectat:", corrected_text)
