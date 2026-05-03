# GSCR Framework Detaliat pentru CIKM 2026

Acest document explică arhitectura tehnică a proiectului GSCR și extensiile implementate pentru a susține o lucrare de tip Full Paper (CIKM).

Se bazează pe documentația existentă din `README.md` și `bench/README.md`, dar adaugă:
- descrierea completă a fluxului runtime,
- justificarea metodologică a reranker-ului conservator,
- protocolul de benchmark pe 3 niveluri,
- ghidul de transformare în secțiuni de articol.

---

## 1) Arhitectura end-to-end

Pipeline-ul principal este `Preprocess -> Detect -> Correct -> Rank -> Feedback`.

### A. Preprocessing
- fișier relevant: `src/preprocess/teprolin_pipeline.py`
- rol: normalizare text, tokenizare, procesare lingvistică (Teprolin + spaCy în ecosistemul proiectului)
- utilizare în API: endpointurile din `app.py` aplică preprocess înainte de generare.

### B. Error Detection (ULMFiT-style classifier)
- fișier relevant: `src/detection/detect.py`
- modelul `HFWrapperULMFiT` întoarce `logits` pentru clasele `{Correct, Incorrect}`
- în `/check` (din `app.py`) probabilitatea `p(Incorrect)` este folosită acum și ca semnal de incertitudine pentru reranking.

### C. Automatic Correction (T5)
- fișier relevant: `src/models.py`
- `generate_corrections(...)`: baseline de generare top-k
- `generate_corrections_with_scores(...)` (implementare nouă): întoarce candidați + scor proxy `avg_logprob` din pașii de generare.

### D. Suggestion Ranking
- fișier relevant: `src/suggestion_ranker.py`
- baseline vechi: scor BLEU față de propoziția originală
- metodă nouă: reranker conservator cu penalizare de edit distance și semnal de încredere.

### E. Active Learning + Feedback
- fișiere relevante: `database/*`, `src/active_learning.py`, endpoint `/feedback` în `app.py`
- feedback-ul utilizatorului este logat și pregătit pentru iterații ulterioare de fine-tuning.

---

## 2) Modificări implementate pentru noutate metodologică

### 2.1 Conservative, uncertainty-aware reranking

Am introdus în `src/suggestion_ranker.py` un scor de tip:

\[
Score(s) = Conf_{det}(x) \cdot \log P_{T5}(s|x) - \lambda \cdot Lev(x,s)
\]

unde:
- `Conf_det(x)` este probabilitatea că intrarea conține eroare (din detector),
- `log P_T5(s|x)` este un proxy din scorurile de generare T5,
- `Lev(x,s)` este distanța Levenshtein (implementare existentă, sensibilă la diacritice în `src/correct_word/levenshtein.py`),
- `\lambda` controlează conservatorismul editărilor.

Impact practic:
- reduce over-correction (rescrieri inutile),
- menține corecturile cu probabilitate mare când detectorul are încredere.

### 2.2 Integrare în API

În `app.py`:
- endpoint `/correct` folosește `generate_corrections_with_scores(...)` + `metric="conservative"`;
- endpoint `/check` calculează `prob_incorrect` din detector și o transmite în ranker.

Această integrare creează o legătură directă între detectare și corectare, nu doar un pipeline secvențial rigid.

---

## 3) Benchmark intern pentru lucrare (GSCR-Benchmark-RO, GEC-only)

### Level 1: In-distribution
- dataset: `GEC/test.txt`
- scop: performanță pe distribuția cunoscută.

### Level 2: Cross-split robustness
- dataset: `GEC/dev.txt`
- scop: verificare robustețe pe split validare, fără date externe.

### Level 3: Stress Test (din GEC)
- generator: `bench/build_level2_stress.py`
- intrare: `GEC/test.txt`
- ieșire: `bench/level3_stress_from_gec.txt`
- perturbări: eliminare diacritice + confuzii agreement-like.

### Evaluare automată
- script: `bench/run_benchmark.py`
- output: `bench/results.json`
- metrici:
  - `exact_match`,
  - `BLEU`,
  - `chrF`,
  - `overcorrection_rate`,
  - `unchanged_rate_on_correct`,
  - `changed_rate`.

### Randare tabele pentru articol
- script: `bench/render_tables.py`
- output:
  - `bench/tables.md`,
  - `bench/tables.tex` (paper-ready).

---

## 4) Ablații incluse (obligatorii pentru argument științific)

În `bench/run_benchmark.py` sunt definite strategiile:
- `t5_top1` (fără reranker),
- `bleu_ranker` (baseline legacy),
- `conservative` (metoda completă),
- `conservative_no_detconf` (fără contribuția detectorului),
- `conservative_no_edit` (fără penalizare de edit distance).

Interpretare recomandată:
- `conservative` trebuie să câștige la over-correction pe L2/L3,
- cu degradare minimă sau nulă la BLEU/chrF față de baseline.

---

## 5) Comentarea framework-ului în articol (secțiuni recomandate)

### Introduction
- problema: GEC robust pentru română pe date limitate, interne;
- limitare curentă: supra-corectare la sisteme standard.

### Method
- descrierea pipeline-ului GSCR;
- formula reranker și rolul fiecărui termen;
- motivarea `lambda_edit` și folosirea confidenței detectorului.

### Benchmark & Experimental Protocol
- L1/L2/L3 cu motivație clară;
- L3 ca set de stres sintetic derivat din `GEC/test.txt`.

### Results
- tabelele generate din `bench/tables.tex`;
- discuție pe trade-off corectare vs conservatorism.

### Error Analysis
- tipuri de erori: diacritice, acord, punctuație, split/merge tokeni;
- exemple de over-correction evitate.

### Limitations
- lipsă benchmark autentic extern (L3 este sintetic),
- posibil gap față de distribuția de erori din producție.

---

## 6) Comenzi utile (reproducibilitate)

0) Antrenează detectorul și exportă checkpoint local:

```bash
python src/detection/train_detector.py --train GEC/train.txt --dev GEC/dev.txt --test GEC/test.txt --out src/detection/content/trained_model_V2_2
```

După acest pas, checkpoint-ul detectorului devine compatibil direct cu:
- `app.py` (endpoint `/check`)
- `bench/run_benchmark.py` (opțiunea `--detector-path`)

1) Generează Level 3 (stress din GEC):

```bash
python bench/build_level2_stress.py --in GEC/test.txt --out bench/level3_stress_from_gec.txt --max 500 --seed 42
```

2) Rulează benchmark complet:

```bash
python bench/run_benchmark.py --model t5-grammar-finetuned --level1 GEC/test.txt --level2 GEC/dev.txt --level3 bench/level3_stress_from_gec.txt --detector-path src/detection/content/trained_model_V2_2 --k 5 --lambda-edit 0.35 --out bench/results.json
```

3) Generează tabelele pentru paper:

```bash
python bench/render_tables.py --in bench/results.json --out-md bench/tables.md --out-tex bench/tables.tex
```

---

## 7) Ce înseamnă „rezultat bun” pentru submit

Pentru argument credibil de Full Paper:
- câștig consistent pe L3 la `overcorrection_rate`,
- menținere performanță lexicală (`BLEU`, `chrF`) aproape de baseline,
- ablații care demonstrează că ambele componente (confidence + edit penalty) contribuie.

Notă metodologică: în această configurație, L3 este sintetic (nu autentic manual), deci concluziile trebuie raportate ca validare internă pe `./GEC`.

Cu această structură, proiectul nu mai este doar „pipeline funcțional”, ci devine o contribuție metodologică + experimentală clară.

