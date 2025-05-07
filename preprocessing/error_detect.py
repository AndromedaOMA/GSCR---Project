import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from grammar_correction.teprolin_pipeline import teprolin_preprocess, extract_teprolin_tokens

# Initialize RoBERTa tokenizer and model for Romanian
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = AutoModelForTokenClassification.from_pretrained("path/to/fine-tuned-roberta", num_labels=2)
model.eval()

def preprocess_for_roberta(text):
    """
    Use Teprolin to preprocess input text, and prepare it for RoBERTa.
    """
    teprolin_result = teprolin_preprocess(text)
    tokenized_data = extract_teprolin_tokens(teprolin_result)

    tokens = []
    for sentence_tokens in tokenized_data:
        tokens.extend([tok["_wordform"] for tok in sentence_tokens])

    return tokens


def predict_grammar_errors(tokens):
    """
    Given preprocessed tokens, return grammar error predictions by RoBERTa.
    """
    # Encode tokens using RoBERTa tokenizer
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    token_predictions = []
    for token, pred in zip(tokens, predictions):
        status = "Incorrect" if pred == 1 else "Correct"
        token_predictions.append((token, status))

    return token_predictions


def main():
    input_text = "Diabetul zaharat se remarca prin valori crescut ale concentratiei glucozei in sange."

    print("Original text:\n", input_text)

    # Preprocess using Teprolin
    tokens = preprocess_for_roberta(input_text)

    print("\nTokens after Teprolin preprocessing:")
    print(tokens)

    # Grammar error prediction
    predictions = predict_grammar_errors(tokens)

    print("\nRoBERTa Grammar Error Detection Results:")
    for token, status in predictions:
        print(f"{token}: {status}")


if __name__ == "__main__":
    main()
