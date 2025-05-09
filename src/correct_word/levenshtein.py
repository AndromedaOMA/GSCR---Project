from pathlib import Path
from typing import List
from symspellpy import SymSpell, Verbosity
import unicodedata

# !! Be aware to not call this dict for distance > 2, error will be thrown
sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

VOCAB_PATH = Path(__file__).resolve().parents[2] / "src" / "correct_word" / "extract" / "corpus.txt"
sym.create_dictionary(
    VOCAB_PATH,
    encoding="utf-8",
    errors="ignore"
)

def is_base_letter_and_diacritic_form(char1: str, char2: str) -> bool:
    """
    Return True if char1 and char2 share the same base letter but differ
    by a diacritic (precomposed or combining form).
    """
    def split_base_and_marks(c: str):
        norm = unicodedata.normalize('NFD', c)
        base = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
        marks = [ch for ch in norm if unicodedata.category(ch).startswith('M')]
        return base, marks

    b1, _ = split_base_and_marks(char1)
    b2, _ = split_base_and_marks(char2)
    # same single‑letter base, different full characters
    return len(b1) == len(b2) == 1 and b1 == b2 and char1 != char2


def levenshtein(w1: str, w2: str) -> float:
    """
    Compute the Levenshtein edit distance between w1 and w2,
    treating a diacritic substitution as cost 0.25 instead of 1
    """
    m, n = len(w1), len(w2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(m):
        curr[0] = i + 1
        for j in range(n):
            sub_cost = 0 if (w1[i] == w2[j] or is_base_letter_and_diacritic_form(w1[i], w2[j])) else 1
            cost_sub = prev[j] + sub_cost
            curr[j+1] = min(
                prev[j+1] + 1,    # deletion
                curr[j]   + 1,    # insertion
                cost_sub          # substitution (0/0.25/1)
            )
        prev, curr = curr, [0] * (n + 1)

    return float(prev[n])

def recommend_corrected_word(
    word: str,
    num_suggestions: int = 5,
    max_plain_dist: int = 2,
    max_diacritic_dist: float = 1.0
) -> List[str]:
    raw = sym.lookup(word, Verbosity.ALL, max_edit_distance=max_plain_dist)
    scored = [(cand.term, levenshtein(word, cand.term)) for cand in raw]
    filtered = [w for w,d in scored if d <= max_diacritic_dist]
    filtered.sort(key=lambda w: levenshtein(word, w))
    return filtered[:num_suggestions]