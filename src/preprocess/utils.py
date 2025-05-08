import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def generate_corrections(text, num_options=5):
    """ Generează multiple opțiuni de corectare pentru un text dat """
    tokenizer = AutoTokenizer.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")
    model = AutoModelForSeq2SeqLM.from_pretrained("BlackKakapo/t5-small-grammar-ro-root")

    prefix = "grammar: "
    input_text = prefix + text
    inputs = tokenizer(input_text, return_tensors="pt")

    outputs = model.generate(**inputs, num_return_sequences=num_options, num_beams=num_options)

    options = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return options