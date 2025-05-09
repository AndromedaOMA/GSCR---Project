import sys
import pathlib
import torch
import re
from transformers import AutoTokenizer, AutoConfig

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from src.preprocess.utils import generate_corrections
from src.preprocess.HFWrapperULMFiT import HFWrapperULMFiT

# TODO - FROM FINE-TUNED MODEL (now it has random weights each time)
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
# model = ULMFiTClassifier("dumitrescustefan/bert-base-romanian-cased-v1").to(device)
config = AutoConfig.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", num_labels=2)
model = HFWrapperULMFiT(config)

# ✅ Light Romanian preprocessing (no Teprolin)
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # remove extra whitespace
    return text

# ✅ Inference function
def predict_on_text(text: str, model, tokenizer, max_length: int = 128) -> str:
    model.eval()

    # Clean and tokenize
    text = clean_text(text)
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    # Move tensors to correct device
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map prediction to label
    label_map = {0: "✅ Correct", 1: "❌ Incorrect"}
    return label_map[predicted_class],predicted_class

# Example usage
examples = [
    "Mam mai gândit că poate voia să țepuiască pe cineva cu carnea aia urât mirositoare.",
    "",
    "L-a proasta satului nu vine nimeni",
    "Situația va rămâne instabilă până l-a dispariția familiei Roanne-Saint-Maurice.",
    "Ea este femeia pe care am văzut-o",
    "M-a  lovit ca un trasnet"
]

for sentence in examples:
    result,decision = predict_on_text(sentence, model, tokenizer)
    print(f"\"{sentence}\"\n→ {result}\n")
    if(decision==1):
      print(f"Corrected: \"{generate_corrections(sentence)[0]}\"")