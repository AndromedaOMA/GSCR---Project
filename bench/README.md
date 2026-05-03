## GSCR internal benchmark (GEC-only)

### Data format
All datasets follow the repo convention used in `GEC/train.txt`, `dev.txt`, `test.txt`:

- Line 1: **target** (correct sentence)
- Line 2: **input** (noisy / incorrect sentence)
- Repeats (even number of non-empty lines)

### GEC-only split protocol (no external/manual data)

We keep everything strictly inside `./GEC` and define benchmark levels as follows:

- **Training set**: `GEC/train.txt`
- **Validation set**: `GEC/dev.txt`
- **Benchmark Level 1 (in-distribution)**: `GEC/test.txt`
- **Benchmark Level 2 (cross-split robustness)**: `GEC/dev.txt`
- **Benchmark Level 3 (stress from existing GEC data)**: generated from `GEC/test.txt` via controlled perturbations

This removes the dependency on `level3_authentic.txt` and keeps evaluation fully reproducible with current repository data.

### Level 3 — Stress (from GEC only)
Generate a stress set from existing GEC targets:

```bash
python bench/build_level2_stress.py --in GEC/test.txt --out bench/level3_stress_from_gec.txt --max 500 --seed 42
```

### Evaluation (Level 1/2/3)
Run the conservative reranker evaluation:

```bash
python bench/eval_gec.py --model t5-grammar-finetuned --data GEC/test.txt --k 5 --lambda-edit 0.35
python bench/eval_gec.py --model t5-grammar-finetuned --data GEC/dev.txt --k 5 --lambda-edit 0.35
python bench/eval_gec.py --model t5-grammar-finetuned --data bench/level3_stress_from_gec.txt --k 5 --lambda-edit 0.35
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
  --level2 GEC/dev.txt \
  --level3 bench/level3_stress_from_gec.txt \
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

All three levels are generated/loaded from existing repository data (`./GEC` + derived stress set).

### Render tables for paper
Convert `bench/results.json` into Markdown and LaTeX tables:

```bash
python bench/render_tables.py --in bench/results.json --out-md bench/tables.md --out-tex bench/tables.tex
```

Generated files:
- `bench/tables.md` (quick inspection / docs)
- `bench/tables.tex` (paper-ready LaTeX tables)

### Notes for scientific reporting
- This protocol is fully **GEC-only** (no external/manual dataset).
- In the paper, describe Level 3 as **synthetic stress**, not authentic real-world data.
- Keep `GEC/test.txt` as final in-distribution benchmark and avoid tuning on it.

