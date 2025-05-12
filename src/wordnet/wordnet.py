import sys
import pathlib
import json
import spacy
from pathlib import Path
from typing import List, Dict

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))

from src.wordnet.literals_lookup import get_synonyms, get_hypernyms, get_hyponyms

# 1) load spaCy Romanian
nlp = spacy.load("ro_core_news_sm")

# 2) load our pre-built inflection index
INFLECTED_INDEX = Path(__file__).parent / "extract" / "inflected_index.json"
_infl_index: Dict[str, List[Dict]] = {}
with INFLECTED_INDEX.open(encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        lemma = rec["lemma"]
        _infl_index.setdefault(lemma, []).append({
            "form": rec["form"],
            "feats": rec["feats"]
        })

def inflect_from_index(lemma: str, desired_feats: Dict[str,str]) -> str:
    """
    Look up the list of all known inflected forms of `lemma`,
    and return the one whose feats exactly match `desired_feats`.
    Fallback: first form with same Number, then the bare lemma.
    """
    candidates = _infl_index.get(lemma, [])
    # exact match
    for ent in candidates:
        if ent["feats"] == desired_feats:
            return ent["form"]
    # partial fallback: same Number
    want_num = desired_feats.get("Number")
    for ent in candidates:
        if ent["feats"].get("Number") == want_num:
            return ent["form"]
    # give up
    return lemma

def related_forms(
    word_form: str
) -> Dict[str, List[str]]:
    doc = nlp(word_form)
    if not doc or not doc[0].has_morph:
        lemma = word_form
        return {
            "input": word_form,
            "lemma": lemma,
            "synonyms": get_synonyms(lemma),
            "hypernyms": get_hypernyms(lemma),
            "hyponyms": get_hyponyms(lemma),
        }

    tok    = doc[0]
    lemma  = tok.lemma_
    feats  = tok.morph.to_dict()

    raw_syn  = get_synonyms(lemma)
    raw_hyp  = get_hypernyms(lemma)
    raw_hyno = get_hyponyms(lemma)

    out = {
        "input":     word_form,
        "lemma":     lemma,
        "synonyms":  [],
        "hypernyms": [],
        "hyponyms":  [],
    }

    for lst, key in ((raw_syn,  "synonyms"),
                     (raw_hyp,  "hypernyms"),
                     (raw_hyno, "hyponyms")):
        for cand in lst:
            out[key].append(inflect_from_index(cand, feats))

    return out

if __name__ == "__main__":
    for w in ["mașină", "mașina", "mașini", "mașinii"]:
        res = related_forms(w)
        print(f"\n── Related for “{w}” (lemma={res['lemma']}):")
        print(" → synonyms :", res["synonyms"][:5])
        print(" → hypernyms:", res["hypernyms"][:5])
        print(" → hyponyms :", res["hyponyms"][:5])