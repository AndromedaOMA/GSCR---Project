import argparse

def load_words(path):
    """Read non-empty, stripped lines from a file into a set."""
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w:
                words.add(w)
    return words

def main():
    parser = argparse.ArgumentParser(
        description="Merge two word-list files without duplicates."
    )
    parser.add_argument(
        "lemma",
        help="Path to lemma.txt",
        nargs="?",
        default="lemma.txt"
    )
    parser.add_argument(
        "inflected",
        help="Path to inflected.txt",
        nargs="?",
        default="inflected.txt"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: corpus.txt)",
        default="corpus.txt"
    )
    args = parser.parse_args()

    # Load and union
    merged = load_words(args.lemma) | load_words(args.inflected)

    # Write out sorted
    with open(args.output, "w", encoding="utf-8") as out:
        for word in sorted(merged):
            out.write(word + "\n")

    print(f"Merged {len(merged)} unique words into {args.output!r}")

if __name__ == "__main__":
    main()