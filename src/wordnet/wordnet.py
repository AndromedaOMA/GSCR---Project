import sys
import pathlib
import spacy
from typing import List, Dict

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
from src.wordnet.literals_lookup import get_synonyms, get_hypernyms, get_hyponyms

nlp = spacy.load("ro_core_news_sm")


def reinflect_noun(lemma: str, feats: Dict[str,str]) -> str:
    num = feats.get("Number")
    det = feats.get("Definite")  # "Def" or None
    gen = feats.get("Gender")    # "Fem" or "Masc"/"Neut"/None

    is_vowel = lemma[-1] in "aeiouăîâ"
    fem_ă    = lemma.endswith("ă")
    stem     = lemma[:-1] if fem_ă else lemma

    # Hard code the forms as it is not present in any downloaded dataset
    if num == "Sing" and det != "Def":
        return lemma

    if num == "Sing" and det == "Def":
        if gen == "Fem" and fem_ă:
            return stem + "a"
        return lemma + ("le" if is_vowel else "ul")

    if num == "Plur" and det != "Def":
        if gen == "Fem" and fem_ă:
            return stem + "i"
        if is_vowel:
            return lemma[:-1] + "e"
        return lemma + "e"

    if num == "Plur" and det == "Def":
        if gen == "Fem" and fem_ă:
            base = stem + "i"
        else:
            base = (lemma[:-1] + "e") if is_vowel else (lemma + "e")
        return base + "le"

    # fallback
    return lemma


def reinflect(cand: str, input_feats: Dict[str,str], input_pos: str) -> str:
    if input_pos != "NOUN":
        return cand  # only nouns supported

    # get gender
    doc2 = nlp(cand)
    if not doc2 or not doc2[0].has_morph:
        gen = None
    else:
        gen = doc2[0].morph.get("Gender")
        gen = gen[0] if gen else None

    feats = {
        "Number":   input_feats.get("Number"),
        "Definite": input_feats.get("Definite"),
        "Gender":   gen or "Masc"  # default to Masc if unknown
    }

    return reinflect_noun(cand, feats)


def related_forms(word_form: str) -> Dict[str, List[str]]:
    doc = nlp(word_form)
    if not doc or not doc[0].has_morph:
        lemma = word_form
        return {
            "input":     word_form,
            "lemma":     lemma,
            "synonyms":  get_synonyms(lemma),
            "hypernyms": get_hypernyms(lemma),
            "hyponyms":  get_hyponyms(lemma),
        }

    tok   = doc[0]
    lemma = tok.lemma_
    feats = tok.morph.to_dict()
    pos   = tok.pos_

    out = {
        "input":     word_form,
        "lemma":     lemma,
        "synonyms":  [],
        "hypernyms": [],
        "hyponyms":  [],
    }

    for fn, key in (
        (get_synonyms,  "synonyms"),
        (get_hypernyms, "hypernyms"),
        (get_hyponyms,  "hyponyms"),
    ):
        for cand in fn(lemma):
            out[key].append(reinflect(cand, feats, pos))

    return out

def get_related_forms(word_form: str) -> str:
    data = related_forms(word_form)
    return data.get("synonyms", [])[:3] + data.get("hypernyms", [])[:1] + data.get("hyponyms", [])[:1]

if __name__ == "__main__":
    # Test example 
    for w in ["mașină", "mașina", "mașini", "mașinii", "mergând"]:
        res = related_forms(w)
        print(f"\n── Related for “{w}” (lemma={res['lemma']}):")
        print(" → synonyms :",  res["synonyms"][:5])
        print(" → hypernyms:", res["hypernyms"][:5])
        print(" → hyponyms :", res["hyponyms"][:5])