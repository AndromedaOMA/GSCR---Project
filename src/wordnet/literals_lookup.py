import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

ROW_XML_PATH = Path(__file__).resolve().parents[2] / "src" / "wordnet" / "extract" / "rown.xml"


_literal_to_synsets: dict[str, set[str]] = defaultdict(set)
_synset_to_literals: dict[str, list[str]] = {}
_synset_to_hypernyms: dict[str, list[str]] = defaultdict(list)


def _load_rown(xml_path: Path = ROW_XML_PATH):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for syn in root.findall("SYNSET"):
        sid = syn.find("ID").text
        # collect all literal forms in this synset
        lits = [lit.text for lit in syn.findall("./SYNONYM/LITERAL")]
        _synset_to_literals[sid] = lits
        # for each literal, note it lives in this synset
        for lit in lits:
            _literal_to_synsets[lit].add(sid)
        # collect hypernym links
        for ilr in syn.findall("ILR"):
            if ilr.find("TYPE").text == "hypernym":
                _synset_to_hypernyms[sid].append(ilr.text)


# load once at import time
_load_rown()


def get_synonyms(word: str) -> list[str]:
    syns = set()
    for sid in _literal_to_synsets.get(word, []):
        for lit in _synset_to_literals.get(sid, []):
            if lit != word:
                syns.add(lit)
    return sorted(syns)


def get_hypernyms(word: str) -> list[str]:
    hypers = set()
    for sid in _literal_to_synsets.get(word, []):
        for hyper_sid in _synset_to_hypernyms.get(sid, []):
            for lit in _synset_to_literals.get(hyper_sid, []):
                hypers.add(lit)
    return sorted(hypers)


# optional: reverse lookup
def get_hyponyms(word: str) -> list[str]:
    hypos = set()
    target_sids = _literal_to_synsets.get(word, set())
    for sid, hyper_list in _synset_to_hypernyms.items():
        if any(h in target_sids for h in hyper_list):
            for lit in _synset_to_literals.get(sid, []):
                hypos.add(lit)
    return sorted(hypos)


if __name__ == "__main__":
    for w in ["mașină"]:
        print(f"\nWord: {w}")
        print(" Synonyms:   ", get_synonyms(w))
        print(" Hypernyms:  ", get_hypernyms(w))
        print(" Hyponyms:   ", get_hyponyms(w))
        