import re
from tqdm import tqdm

def extract_form_utf8_general(sql_path, out_path):
    # This regex matches:
    #   (   <digits> , 'first' , 'second' , 'third' ,
    # and captures the third quoted field in group 3.
    pattern = re.compile(
        r"""
        \(\s*                 # opening paren
        \d+\s*,\s*            # id,
        '((?:\\'|[^'])*)'\s*,\s*   # 1st field (we don't use it)
        '((?:\\'|[^'])*)'\s*,\s*   # 2nd field (we don't use it)
        '((?:\\'|[^'])*)'          # 3rd field = formUtf8General → group(3)
        """,
        re.UNICODE | re.VERBOSE
    )

    seen = set()
    # 'replace' so we don't silently drop bytes
    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, desc="Scanning", unit="lines"):
            for m in pattern.finditer(line):
                word = m.group(3).replace("\\'", "'")
                seen.add(word)

    with open(out_path, 'w', encoding='utf-8') as out:
        for w in sorted(seen):
            out.write(w + "\n")

    print(f"Extracted {len(seen)} unique forms into {out_path!r}")

if __name__ == "__main__":
    extract_form_utf8_general("inflected.sql", "inflected.txt")
    extract_form_utf8_general("lemma.sql", "lemma.txt")