"""
Turn retrieval_metrics.json into a readable Markdown evaluation report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def metric_table(metrics: dict[str, float]) -> str:
    names = ["recall@1", "recall@3", "recall@5", "recall@10",
             "precision@5", "mrr", "hit_rate@5"]
    lines = ["| Metric | Score |", "|---|---:|"]
    for name in names:
        if name in metrics:
            lines.append(f"| {name} | {metrics[name]:.4f} |")
    return "\n".join(lines)


def grouped_table(groups: dict[str, Any]) -> str:
    lines = ["| Group | Queries | Recall@5 | MRR |", "|---|---:|---:|---:|"]
    for name, data in groups.items():
        m = data["metrics"]
        lines.append(
            f"| {name} | {data['query_count']} | "
            f"{m.get('recall@5', 0):.4f} | {m.get('mrr', 0):.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("retrieval_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieval_report.md"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    ds = data["dataset"]
    latency = data.get("latency_ms", {})

    report = f"""# Egyptian Civil Code — Retrieval Evaluation

## Evaluation setup

- Evaluated queries: **{ds['evaluated_queries']}**
- Failed queries: **{ds['failed_queries']}**
- Relevance unit: **{ds['relevance_unit']}**
- Cutoffs: **{', '.join(map(str, ds['k_values']))}**
- Aggregation: macro-average across labeled queries

## Overall retrieval performance

{metric_table(data['overall'])}

## Performance by query type

{grouped_table(data.get('by_type', {}))}

## Performance by language

{grouped_table(data.get('by_language', {}))}

## Latency

- Mean: {latency.get('mean')} ms
- Median: {latency.get('median')} ms
- P95: {latency.get('p95')} ms

## Interpretation notes

"""
    for note in data.get("notes", []):
        report += f"- {note}\n"

    report += """
## Recommended next steps

1. Inspect queries with low Recall@5 and MRR in `retrieval_per_query.csv`.
2. Review article-lookup queries separately; they should exercise the direct article
   metadata lookup path, not only dense semantic retrieval.
3. Check follow-up queries with their conversation history. A follow-up evaluated
   without history is not a valid test of contextual retrieval.
4. Compare retrieval changes on this same fixed test set: baseline dense retrieval,
   hybrid retrieval, and reranking. Change one variable at a time.
5. Do not treat retrieval metrics as answer-quality metrics. Evaluate answer
   correctness, grounding, and citation accuracy in a separate stage.
"""

    args.output.write_text(report, encoding="utf-8")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
