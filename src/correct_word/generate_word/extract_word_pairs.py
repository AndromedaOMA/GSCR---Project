import os
import random
import string
import pandas as pd

DATA_DIR     = r"D:\GSCR---Project\GEC"
AUG_PER_PAIR = 3

DIACRITIC_MAP = {
    "a": ["ă", "â"],
    "A": ["Ă", "Â"],
    "i": ["î"],
    "I": ["Î"],
    "s": ["ș"],
    "S": ["Ș"],
    "t": ["ț"],
    "T": ["Ț"],
}

def extract_word_pairs(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) % 2 == 0, f"{path} must have an even number of non-empty lines"

    out = []
    for i in range(0, len(lines), 2):
        corr = lines[i]
        inc  = lines[i+1]
        tc, ti = corr.split(), inc.split()
        if len(tc) != len(ti):
            continue
        diffs = [(j, ti[j], tc[j]) for j in range(len(ti)) if ti[j] != tc[j]]
        if len(diffs) == 1:
            _, wrong, right = diffs[0]
            out.append({"wrong": wrong, "correct": right, "sent_corr": corr})
    return out

def perturb_token(token: str) -> str:
    if not token:
        return token

    ops = ["delete", "swap", "replace", "diacritic"]
    op  = random.choice(ops)
    chars = list(token)

    if op == "delete" and len(chars) > 1:
        del chars[random.randrange(len(chars))]

    elif op == "swap" and len(chars) > 1:
        i = random.randrange(len(chars)-1)
        chars[i], chars[i+1] = chars[i+1], chars[i]

    elif op == "replace":
        i = random.randrange(len(chars))
        cand = random.choice(string.ascii_lowercase)
        if chars[i].isupper():
            cand = cand.upper()
        chars[i] = cand

    elif op == "diacritic":
        idxs = [i for i, ch in enumerate(chars) if ch in DIACRITIC_MAP or ch.lower() in DIACRITIC_MAP]
        if idxs:
            i = random.choice(idxs)
            base = chars[i]
            choices = DIACRITIC_MAP.get(base, DIACRITIC_MAP.get(base.lower(), []))
            if choices:
                chars[i] = random.choice(choices)

    return "".join(chars)

def augment_pairs(pairs, aug_per_pair=AUG_PER_PAIR):
    augmented = []
    for rec in pairs:
        corr_tok = rec["correct"]
        for _ in range(aug_per_pair):
            noisy = perturb_token(corr_tok)
            if noisy == corr_tok:
                continue
            toks = rec["sent_corr"].split()
            try:
                idx = toks.index(corr_tok)
            except ValueError:
                continue
            toks[idx] = noisy
            augmented.append({"wrong": noisy, "correct": corr_tok})
    return augmented

if __name__ == "__main__":
    paths = {
        "train": os.path.join(DATA_DIR, "train.txt"),
        "dev":   os.path.join(DATA_DIR, "dev.txt"),
        "test":  os.path.join(DATA_DIR, "test.txt"),
    }
    for p in paths.values():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Could not find {p}")

    train_pairs = extract_word_pairs(paths["train"])
    dev_pairs   = extract_word_pairs(paths["dev"])
    test_pairs  = extract_word_pairs(paths["test"])

    print(f"Original examples → train={len(train_pairs)}, dev={len(dev_pairs)}, test={len(test_pairs)}")

    aug = augment_pairs(train_pairs, AUG_PER_PAIR)
    print(f"  → generated {len(aug)} augmented examples")

    full = train_pairs + dev_pairs + test_pairs
    # drop the sentence context column and just keep wrong/correct
    df   = pd.DataFrame(full)[["wrong", "correct"]]

    # tack on the augmented ones
    df_aug = pd.DataFrame(aug)
    final  = pd.concat([df, df_aug], ignore_index=True)

    print(f"→ Total rows for modeling (wrong,correct): {len(final)}")
    print(final.head(10))

    out_csv = os.path.join(DATA_DIR, "word_correction_pairs_augmented.csv")
    final.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"→ Saved CSV to {out_csv}")
