"""Paired significance test between two eval runs (same split, same examples).

Two systems evaluated through the frozen harness score the *same* examples, so the honest
comparison is **paired**: for each example, take the per-example metric difference and ask whether
its mean is distinguishable from zero. Paired tests are far tighter than comparing two independent
proportions, because the systems agree on most examples and that shared variance cancels.

Reports, per metric:
  - the mean difference (B - A) and its bootstrap 95% CI (resampling examples with replacement);
  - a two-sided p-value from an exact-style sign-flip permutation test on the paired differences
    (the null: the sign of each example's difference is arbitrary);
  - the win/loss/tie counts, so a mean shift driven by a handful of examples is visible.

Usage:
    python scripts/significance.py \
        --a results/metrics/late_interaction_ft_novel_novel_premises.json \
        --b results/metrics/late_interaction_ft_novel_weighted_novel_premises.json

Reads only the `examples[]` records written by eval/evaluate.py; changes no metric and touches no
model. Seeded, so the reported CI/p-value are reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_per_example(path: str) -> tuple[dict, dict[str, dict[str, float]]]:
    """Return (provenance, {eid: {metric: value}}) from an eval metrics JSON."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    prov = blob.get("provenance", {})
    by_eid = {r["eid"]: r["metrics"] for r in blob["examples"]}
    return prov, by_eid


def paired_deltas(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]],
                  metric: str) -> np.ndarray:
    """Per-example (B - A) differences over the examples both runs scored, in a stable eid order."""
    eids = sorted(set(a) & set(b))
    if not eids:
        raise ValueError("the two runs share no example ids — are they the same split?")
    return np.array([b[e][metric] - a[e][metric] for e in eids], dtype=np.float64)


def bootstrap_ci(d: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (resample examples, not systems)."""
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def permutation_p(d: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p-value for H0: the paired differences are symmetric at 0.

    Only nonzero differences carry information (ties contribute 0 under any sign), so we flip signs
    on those. The +1/(n+1) correction keeps p strictly positive (never claims p = 0).
    """
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0
    observed = abs(nz.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(nz)))
    null = np.abs((signs * nz).mean(axis=1))
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def compare(path_a: str, path_b: str, metrics: list[str], n_boot: int = 10000,
            n_perm: int = 10000, seed: int = 42) -> list[dict]:
    prov_a, a = load_per_example(path_a)
    prov_b, b = load_per_example(path_b)
    if prov_a.get("split") and prov_b.get("split") and prov_a["split"] != prov_b["split"]:
        raise ValueError(
            f"refusing to compare different splits: {prov_a['split']!r} vs {prov_b['split']!r}"
        )

    rows = []
    for metric in metrics:
        d = paired_deltas(a, b, metric)
        rng = np.random.default_rng(seed)          # same seed per metric -> reproducible
        lo, hi = bootstrap_ci(d, n_boot, rng)
        p = permutation_p(d, n_perm, rng)
        rows.append({
            "metric": metric,
            "n": len(d),
            "mean_a": float(np.mean([a[e][metric] for e in sorted(set(a) & set(b))])),
            "mean_b": float(np.mean([b[e][metric] for e in sorted(set(a) & set(b))])),
            "delta": float(d.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p,
            "wins": int(np.sum(d > 0)),
            "losses": int(np.sum(d < 0)),
            "ties": int(np.sum(d == 0)),
            "significant": bool(lo > 0 or hi < 0),   # CI excludes zero
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="baseline run's metrics JSON")
    ap.add_argument("--b", required=True, help="comparison run's metrics JSON")
    ap.add_argument("--metrics", nargs="+", default=["R@1", "R@10", "MRR", "nDCG@10"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="optional JSON path to save the table")
    args = ap.parse_args()

    rows = compare(args.a, args.b, args.metrics, args.n_boot, args.n_perm, args.seed)

    print(f"\nA (baseline): {Path(args.a).name}")
    print(f"B (compare) : {Path(args.b).name}")
    print(f"\n{'metric':<9} {'A':>8} {'B':>8} {'delta':>8} {'95% CI':>18} {'p':>8}  "
          f"{'W/L/T':>16}  verdict")
    print("-" * 96)
    for r in rows:
        ci = f"[{r['ci_low']*100:+.2f},{r['ci_high']*100:+.2f}]"
        wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
        verdict = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"{r['metric']:<9} {r['mean_a']*100:>7.2f}% {r['mean_b']*100:>7.2f}% "
              f"{r['delta']*100:>+7.2f} {ci:>18} {r['p_value']:>8.4f}  {wlt:>16}  {verdict}")
    print(f"\nPaired over n={rows[0]['n']} examples; percentile bootstrap + sign-flip permutation, "
          f"seed={args.seed}. Deltas/CI in percentage points.\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
