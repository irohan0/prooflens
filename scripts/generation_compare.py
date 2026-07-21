"""Compare Phase-21 tactic-generation conditions and test the differences (paired).

Turns the per-condition result JSONs into the table the write-up needs, and — because every
condition scores the **same examples through the same fixed generator** — tests each difference
with the *paired* bootstrap + sign-flip permutation already used in Phase 19, rather than eyeballing
two percentages. Reuses `scripts/significance.py` outright so generation and retrieval claims are
tested by identical statistics.

    python scripts/generation_compare.py --runs results/metrics/gen_*_novel_premises.json

    # explicit head-to-head (the thesis comparison: ours vs the matched control)
    python scripts/generation_compare.py --runs results/metrics/gen_*_novel_premises.json \
        --vs gen_ft_sv_novel gen_ft_li_novel

By default every condition is compared against the **no-premises floor**, which is the honest
first question: does this retriever help the generator at all? The baseline is auto-detected as the
run whose `premise_condition.source == "none"`.

Reads only persisted JSON — no model, no GPU, no re-generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from significance import compare  # noqa: E402

DEFAULT_METRICS = ["match@1", "match@8", "premise_name@1", "premise_name@8"]


def load_run(path: str) -> dict:
    """Load one generation results JSON into a compact summary record."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    prov = blob.get("provenance", {})
    examples = blob.get("examples", [])
    in_ctx = [r.get("n_premises_in_context", 0) for r in examples]
    return {
        "path": path,
        "name": prov.get("config_name", Path(path).stem),
        "split": prov.get("split"),
        "source": prov.get("premise_condition", {}).get("source"),
        "retriever": prov.get("premise_condition", {}).get("retriever_config_name"),
        "n": prov.get("n_examples", len(examples)),
        "is_full_run": prov.get("is_full_run"),
        "metrics": blob.get("metrics", {}),
        "mean_in_context": (sum(in_ctx) / len(in_ctx)) if in_ctx else 0.0,
        "seconds_per_example": prov.get("seconds_per_example"),
    }


def find_baseline(runs: list[dict]) -> dict | None:
    """The no-premises floor, if one of the runs is it."""
    for r in runs:
        if r["source"] == "none":
            return r
    return None


def check_comparable(runs: list[dict]) -> None:
    """Refuse to build a table from runs that are not measuring the same thing.

    A silently-mixed table (different splits, or a pilot subset next to a full run) is exactly the
    kind of plausible-looking artefact this project treats as a bug, so it fails loudly.
    """
    splits = {r["split"] for r in runs}
    if len(splits) > 1:
        raise ValueError(
            f"runs span multiple splits {sorted(splits)} — compare one split at a time"
        )
    ns = {r["n"] for r in runs}
    if len(ns) > 1:
        raise ValueError(
            f"runs cover different example counts {sorted(ns)} — a subset (--limit) run cannot be "
            "compared against a full run"
        )
    partial = [r["name"] for r in runs if r["is_full_run"] is False]
    if partial:
        print(f"  WARNING: subset (--limit) runs, NOT full-split results: {', '.join(partial)}\n")


def print_absolute_table(runs: list[dict], metrics: list[str]) -> None:
    width = max(len(r["name"]) for r in runs) + 2
    header = f"{'condition':<{width}}" + "".join(f"{m:>16}" for m in metrics)
    header += f"{'prem in ctx':>13}"
    print(header)
    print("-" * len(header))
    for r in sorted(runs, key=lambda x: x["metrics"].get(metrics[0], 0)):
        row = f"{r['name']:<{width}}"
        for m in metrics:
            v = r["metrics"].get(m)
            row += f"{v * 100:>15.2f}%" if v is not None else f"{'-':>16}"
        row += f"{r['mean_in_context']:>13.1f}"
        print(row)


def verdict(row: dict, alpha: float = 0.05) -> str:
    """Require the bootstrap CI to exclude zero **AND** the permutation p to clear alpha.

    `significance.py` sets its `significant` flag from the CI alone. On small effects the two can
    disagree — the Phase-21 dry run produced delta +1.67pp with CI [+0.33,+3.33] (excludes zero)
    but p = 0.0625 — and printing "SIGNIFICANT" for a result that fails its own permutation test is
    exactly the kind of overclaim this project refuses. Disagreement is reported as `borderline`
    rather than resolved in our favour.
    """
    ci_excludes_zero = bool(row["significant"])
    p_clears = row["p_value"] < alpha
    if ci_excludes_zero and p_clears:
        return "SIGNIFICANT"
    if ci_excludes_zero or p_clears:
        return "borderline"
    return "ns"


def print_paired(a: dict, b: dict, metrics: list[str], n_boot: int, n_perm: int,
                 seed: int) -> list[dict]:
    rows = compare(a["path"], b["path"], metrics, n_boot, n_perm, seed)
    print(f"\n  {b['name']}  vs  {a['name']}  (paired, n={rows[0]['n']})")
    print(f"  {'metric':<16}{'base':>9}{'this':>9}{'delta':>9}{'95% CI':>20}{'p':>9}  verdict")
    print("  " + "-" * 86)
    for r in rows:
        ci = f"[{r['ci_low'] * 100:+.2f},{r['ci_high'] * 100:+.2f}]"
        r["verdict"] = verdict(r)
        print(f"  {r['metric']:<16}{r['mean_a'] * 100:>8.2f}%{r['mean_b'] * 100:>8.2f}%"
              f"{r['delta'] * 100:>+9.2f}{ci:>20}{r['p_value']:>9.4f}  {r['verdict']}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="generation results JSONs")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--baseline", default=None,
                    help="config_name to compare against (default: the source='none' run)")
    ap.add_argument("--vs", nargs=2, action="append", metavar=("BASE", "COMPARE"), default=[],
                    help="explicit head-to-head, repeatable "
                         "(e.g. --vs gen_ft_sv_novel gen_ft_li_novel)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="optional JSON path for the full comparison")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.runs]
    if not runs:
        raise SystemExit("no runs given")
    check_comparable(runs)
    by_name = {r["name"]: r for r in runs}

    split = runs[0]["split"]
    print(f"\nPhase 21 — tactic generation | split: {split} | n={runs[0]['n']}")
    print("Generator FIXED across all conditions; only the context premises differ.\n")
    print_absolute_table(runs, args.metrics)

    print("\n  match@k     = generated tactic is textually identical to the human's (LOWER bound)")
    print("  premise_name@k = generated tactic NAMES a gold premise (UPPER bound)")
    print("  Neither is a proof-success rate.")

    baseline = by_name.get(args.baseline) if args.baseline else find_baseline(runs)
    out_rows: dict[str, list[dict]] = {}

    if baseline is not None:
        print(f"\n{'=' * 92}\nPAIRED vs the floor ({baseline['name']}) "
              "— does this retriever help at all?")
        for r in runs:
            if r["name"] != baseline["name"]:
                out_rows[f"{baseline['name']}->{r['name']}"] = print_paired(
                    baseline, r, args.metrics, args.n_boot, args.n_perm, args.seed
                )
    else:
        print("\n(no source='none' run supplied — skipping the floor comparison)")

    for base_name, comp_name in args.vs:
        if base_name not in by_name or comp_name not in by_name:
            raise SystemExit(f"--vs {base_name} {comp_name}: unknown run name(s)")
        print(f"\n{'=' * 92}\nHEAD-TO-HEAD")
        out_rows[f"{base_name}->{comp_name}"] = print_paired(
            by_name[base_name], by_name[comp_name], args.metrics,
            args.n_boot, args.n_perm, args.seed
        )

    print(f"\nPaired percentile bootstrap + sign-flip permutation, seed={args.seed}. "
          "Deltas/CI in percentage points.\n")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"runs": runs, "comparisons": out_rows}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
