import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Higher is better for all these metrics in our report.
METRIC_ORDER = [
    "exact_match",
    "bleu",
    "chrf",
    "overcorrection_rate",
    "unchanged_rate_on_correct",
    "changed_rate",
]

METRIC_LABEL = {
    "exact_match": "ExactMatch",
    "bleu": "BLEU",
    "chrf": "chrF",
    "overcorrection_rate": "OverCorr",
    "unchanged_rate_on_correct": "Unchanged@Correct",
    "changed_rate": "ChangedRate",
}


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(level_obj: dict) -> List[Tuple[str, Dict[str, float]]]:
    metrics = level_obj.get("metrics", {})
    rows: List[Tuple[str, Dict[str, float]]] = []
    for strategy, vals in metrics.items():
        row = {m: float(vals.get(m, 0.0)) for m in METRIC_ORDER}
        rows.append((strategy, row))
    return rows


def best_by_metric(rows: List[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
    out = {}
    if not rows:
        return out
    for m in METRIC_ORDER:
        out[m] = max(r[m] for _, r in rows)
    return out


def fmt(v: float) -> str:
    return f"{v:.4f}"


def md_for_level(level_name: str, level_obj: dict) -> str:
    if level_obj.get("status") == "missing_dataset":
        return f"### {level_name}\n\nDataset missing: `{level_obj.get('path', '')}`\n"

    rows = collect_rows(level_obj)
    best = best_by_metric(rows)

    header = "| Strategy | " + " | ".join(METRIC_LABEL[m] for m in METRIC_ORDER) + " |"
    sep = "|---|" + "|".join("---" for _ in METRIC_ORDER) + "|"
    lines = [f"### {level_name}", "", header, sep]

    for strategy, vals in rows:
        cells = [strategy]
        for m in METRIC_ORDER:
            text = fmt(vals[m])
            if vals[m] == best.get(m):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def latex_for_level(level_name: str, level_obj: dict) -> str:
    if level_obj.get("status") == "missing_dataset":
        return f"% {level_name}: missing dataset at {level_obj.get('path', '')}\n"

    rows = collect_rows(level_obj)
    best = best_by_metric(rows)

    cols = "l" + "r" * len(METRIC_ORDER)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Benchmark results for {latex_escape(level_name)}}}",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        "Strategy & " + " & ".join(METRIC_LABEL[m] for m in METRIC_ORDER) + " \\\\",
        "\\hline",
    ]

    for strategy, vals in rows:
        cells = [latex_escape(strategy)]
        for m in METRIC_ORDER:
            text = fmt(vals[m])
            if vals[m] == best.get(m):
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def render(report: dict) -> Tuple[str, str]:
    results = report.get("results", {})
    level_names = list(results.keys())

    md_parts = ["## Benchmark Tables", ""]
    tex_parts = ["% Auto-generated from bench/results.json", ""]

    for level in level_names:
        level_obj = results[level]
        md_parts.append(md_for_level(level, level_obj))
        tex_parts.append(latex_for_level(level, level_obj))

    return "\n".join(md_parts).strip() + "\n", "\n".join(tex_parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="bench/results.json", help="Input JSON report path")
    ap.add_argument("--out-md", default="bench/tables.md", help="Output Markdown table path")
    ap.add_argument("--out-tex", default="bench/tables.tex", help="Output LaTeX table path")
    args = ap.parse_args()

    report_path = Path(args.in_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = load_report(report_path)
    md_text, tex_text = render(report)

    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text, encoding="utf-8")
    out_tex.write_text(tex_text, encoding="utf-8")

    print(f"Wrote {out_md}")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()

