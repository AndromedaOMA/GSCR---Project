## GSCR internal benchmark (draft)

### Data format
All datasets follow the repo convention used in `GEC/train.txt`, `dev.txt`, `test.txt`:

- Line 1: **target** (correct sentence)
- Line 2: **input** (noisy / incorrect sentence)
- Repeats (even number of non-empty lines)

### Level 1 — In-distribution
Use the existing `GEC/test.txt`.

### Level 2 — Stress (diacritics + agreement)
Generate a stress set from gold targets:

```bash
python bench/build_level2_stress.py --in GEC/test.txt --out bench/level2_stress.txt --max 500 --seed 42
```

### Evaluation (Level 1/2)
Run the conservative reranker evaluation:

```bash
python bench/eval_gec.py --model t5-grammar-finetuned --data GEC/test.txt --k 5 --lambda-edit 0.35
python bench/eval_gec.py --model t5-grammar-finetuned --data bench/level2_stress.txt --k 5 --lambda-edit 0.35
```

Reported metrics:
- `exact_match`
- `BLEU` (sacrebleu)
- `chrF` (sacrebleu)
- `overcorrection_rate`: fraction of already-correct inputs that the system changes
- `unchanged_rate_on_correct`
- `changed_rate`: how often the model edits any sentence

### Full benchmark runner (L1/L2/L3 + ablations)
Single command to produce a JSON report with all strategies:

```bash
python bench/run_benchmark.py \
  --model t5-grammar-finetuned \
  --level1 GEC/test.txt \
  --level2 bench/level2_stress.txt \
  --level3 bench/level3_authentic.txt \
  --detector-path src/detection/content/trained_model_V2_2 \
  --k 5 \
  --lambda-edit 0.35 \
  --out bench/results.json
```

Default strategies included in the report:
- `t5_top1` (no reranker)
- `bleu_ranker` (legacy baseline)
- `conservative` (full proposal)
- `conservative_no_detconf` (ablation)
- `conservative_no_edit` (ablation)

If `level3_authentic.txt` is missing, the runner still works and marks it as missing.

### Render tables for paper
Convert `bench/results.json` into Markdown and LaTeX tables:

```bash
python bench/render_tables.py --in bench/results.json --out-md bench/tables.md --out-tex bench/tables.tex
```

Generated files:
- `bench/tables.md` (quick inspection / docs)
- `bench/tables.tex` (paper-ready LaTeX tables)

### Level 3 — Authentic gold (manual)
Recommended structure for `bench/level3_authentic.txt`:

- 250 sentences from real sources (Reddit/Facebook/news/student texts)
- at least 2 annotators per sentence
- keep adjudicated target as the final line-1 target, and raw noisy as line-2 input

