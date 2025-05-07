import requests
import spacy

nlp = spacy.load("ro_core_news_sm")

TEPROLIN_ENDPOINT = "http://localhost:5000/process"

def teprolin_preprocess(text):
    payload = {
        "text": text,
        # operations for the pipeline
        "exec": "diacritics-restoration,text-normalization,sentence-splitting,tokenization,pos-tagging"
    }

    resp = requests.post(TEPROLIN_ENDPOINT, data=payload)
    resp.raise_for_status()
    return resp.json()

def extract_teprolin_sentences(teprolin_result):
    return teprolin_result["teprolin-result"].get("sentences", [])

def extract_teprolin_tokens(teprolin_result):
    #  - tokenized - is a list of lists
    tokenized_data = teprolin_result["teprolin-result"].get("tokenized", [])
    return tokenized_data

def spacy_extra_processing(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    morphological_info = []
    for token in doc:
        morphological_info.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_
        })

    return entities, morphological_info

def main():
    text = "Diabetul zaharat se remarca prin valori crescut ale concentratiei glucozei in sange."

    print("Original text:\n", text)

    teprolin_result = teprolin_preprocess(text)

    corrected_text = teprolin_result["teprolin-result"]["text"]
    print("\nCorrected text (diacritics, normalization):\n", corrected_text)

    sentences = extract_teprolin_sentences(teprolin_result)
    print("\nDetected Sentences (TEPROLIN):")
    for i, sent in enumerate(sentences, start=1):
        print(f"  {i}. {sent}")

    tokenized_data = extract_teprolin_tokens(teprolin_result)
    # tokenized_data is a list of lists of tokens
    print("\nTokenization & POS (TEPROLIN):")
    for i, sentence_tokens in enumerate(tokenized_data, start=1):
        print(f"\nSentence {i} tokens:")
        for tok in sentence_tokens:
            word = tok["_wordform"]
            pos = tok["_ctg"]   # POS from TEPROLIN
            lemma = tok["_lemma"]
            print(f"  {word} ({pos}, lemma={lemma})")

    # spaCy processing
    print("\n--- Additional spaCy Processing ---")
    entities, morph_info = spacy_extra_processing(corrected_text)
    print("Named Entities (spaCy):", entities)
    print("\nMorphological Info (spaCy):")
    for m in morph_info:
        print(f"  {m}")

if __name__ == "__main__":
    main()
