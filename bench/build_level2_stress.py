import argparse
import random
import unicodedata
from typing import List, Tuple


def load_targets(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected even number of lines in {path}")
    # In repo convention, target(correct) is the first line of each pair.
    return [lines[i] for i in range(0, len(lines), 2)]


def strip_diacritics(s: str) -> str:
    # Works for both precomposed and combining marks.
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def corrupt_agreement(s: str, rng: random.Random) -> str:
    """
    Lightweight, deterministic-ish stressor for Romanian agreement.

    This is intentionally simple (paper-friendly): it injects a small number of
    common agreement-like errors without requiring a full morpho analyzer.
    """
    toks = s.split()
    if not toks:
        return s

    # Small set of common confusions / typos around auxiliaries & clitics.
    swaps = {
        "a": "au",
        "au": "a",
        "sunt": "este",
        "este": "sunt",
        "sa": "s-a",
        "s-a": "sa",
        "i": "îi",
        "îi": "i",
    }

    idxs = list(range(len(toks)))
    rng.shuffle(idxs)
    edits = 0
    for i in idxs:
        t = toks[i]
        key = t.lower()
        if key in swaps:
            repl = swaps[key]
            # Preserve capitalization when the token starts the sentence.
            if t[:1].isupper():
                repl = repl[:1].upper() + repl[1:]
            toks[i] = repl
            edits += 1
        if edits >= 2:
            break
    return " ".join(toks)


def make_level2_pairs(
    targets: List[str],
    *,
    seed: int = 42,
    max_items: int = 500,
    p_agreement: float = 0.6,
) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    items = targets[:]
    rng.shuffle(items)
    items = items[: min(max_items, len(items))]

    pairs: List[Tuple[str, str]] = []
    for tgt in items:
        src = strip_diacritics(tgt)
        if rng.random() < p_agreement:
            src = corrupt_agreement(src, rng)
        pairs.append((src, tgt))
    return pairs


def write_pairs(pairs: List[Tuple[str, str]], out_path: str) -> None:
    # Keep the repo convention: target(correct) line first, then input(incorrect)
    with open(out_path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(tgt.strip() + "\n")
            f.write(src.strip() + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to paired dataset (e.g. GEC/test.txt)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output path for Level-2 stress dataset")
    ap.add_argument("--max", dest="max_items", type=int, default=500, help="Max items to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-agreement", type=float, default=0.6)
    args = ap.parse_args()

    targets = load_targets(args.in_path)
    pairs = make_level2_pairs(
        targets,
        seed=args.seed,
        max_items=args.max_items,
        p_agreement=args.p_agreement,
    )
    write_pairs(pairs, args.out_path)
    print(f"Wrote {len(pairs)} pairs to {args.out_path}")


if __name__ == "__main__":
    main()

