import json
from pathlib import Path
import spacy
from tqdm import tqdm

INFLECTED_TXT = Path(__file__).resolve().parents[2] / "correct_word" / "extract" / "corpus.txt"
INDEX_NDJSON  = Path("inflected_index.json")

nlp = spacy.load("ro_core_news_sm")

with INFLECTED_TXT.open(encoding="utf-8") as fin, \
     INDEX_NDJSON.open("w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Indexing forms", unit=" lines"):
        form = line.strip()
        if not form:
            continue

        doc = nlp(form)
        if not doc or not doc[0].has_morph:
            continue

        tok   = doc[0]
        lemma = tok.lemma_
        feats = tok.morph.to_dict()

        record = {
            "lemma": lemma,
            "form":  form,
            "feats": feats
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")