## Prerequisites

```bash
git clone git@github.com:dexonline/dexonline.git src/dexonline
mkdir db
wget -O ./db/dex-database.sql.gz https://dexonline.ro/static/download/dex-database.sql.gz
```

Using winrar, extract the .gz file (will have a 1.2GB file)

## Extraction

From what I've seen, tabel "Lexeme" and "InflectedForm" have valid words. The rest are useless for our use case (extracting all valid words in romanian)

Manually copy the ```INSERT INTO``` lines into different ```.sql``` files for a faster extraction:
- Lexeme inserts into ```lemma.sql```
- InflectedForm inserts into ```inflected.sql```

Run ```extract.py``` and then ```merge.py```