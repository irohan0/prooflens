"""Aggregate a system's per-seed eval JSONs into mean ± std — the Phase-25 variance report.

Give it the eval JSONs for ONE system on ONE split (the seed-42 base run plus its seed variants) and
it reports each metric's mean and sample standard deviation across seeds, so the headline crossover
can be stated as mean ± std rather than a single-seed point estimate.

    python scripts/aggregate_seeds.py --label "FT-LI OFF (novel)" \
        --files results/metrics/late_interaction_ft_novel_novel_premises.json \
                results/metrics/late_interaction_ft_novel_s1_novel_premises.json \
                results/metrics/late_interaction_ft_novel_s2_novel_premises.json ...

Reads only the aggregate `metrics` block of each JSON; touches no model. Sample std (ddof=1).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_METRICS = ["R@1", "R@10", "MRR", "nDCG@10", "MAP"]


def aggregate(files: list[str], metrics: list[str]) -> dict[str, tuple[float, float, int]]:
    """Return {metric: (mean, std, n)} across the given eval JSONs (std is sample/ddof=1)."""
    collected: dict[str, list[float]] = defaultdict(list)
    for f in files:
        m = json.loads(Path(f).read_text(encoding="utf-8"))["metrics"]
        for k in metrics:
            if k in m and m[k] is not None:
                collected[k].append(m[k])
    out: dict[str, tuple[float, float, int]] = {}
    for k, vals in collected.items():
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[k] = (mean, std, len(vals))
    return out


def format_report(label: str, agg: dict[str, tuple[float, float, int]], metrics: list[str]) -> str:
    lines = [f"## {label}"]
    for k in metrics:
        if k in agg:
            mean, std, n = agg[k]
            lines.append(f"  {k:<9} {mean * 100:6.2f} ± {std * 100:4.2f}   (n={n})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mean ± std of a system's per-seed eval JSONs.")
    ap.add_argument("--files", nargs="+", required=True, help="seed-42 base + variant JSONs")
    ap.add_argument("--label", default="system", help="label for the report block")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="metric keys to average")
    args = ap.parse_args()

    agg = aggregate(args.files, args.metrics)
    print(format_report(args.label, agg, args.metrics))
    print(f"\n> metrics as percent; sample std (ddof=1) over {len(args.files)} seeds.")


if __name__ == "__main__":
    main()
