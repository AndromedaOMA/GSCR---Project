from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Încarcă tokenizer-ul și modelul
tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")
model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")

if __name__ == "__main__":
    # Textul de corectat
    sent = ["Miar placea sa merg la munca", "ma doar capul foarrte tare", "Mănanc mazăre cu maree pofta", "Am bafta la bani"]
    prefix = "grammar: "
    for text in sent:
        example = prefix + text

        # Tokenizare
        inputs = tokenizer(example, return_tensors="pt")
        # Generare corectare
        outputs = model.generate(**inputs)
        # Decodificare rezultat
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Initial text: {text}\nCorrected text: {corrected_text}\n")
