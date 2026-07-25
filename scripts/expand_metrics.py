"""Recompute an *extended* metric table from saved eval records — OFFLINE, no model, no GPU.

Every eval JSON written by `eval/evaluate.py` persists, per example, the full top-`retrieve_k`
ranking (`retrieved`) and the gold set (`gold`). Because every @k metric is a pure function of those
two lists, ANY rank-based metric for k <= retrieve_k can be recomputed after the fact without
re-running retrieval — no checkpoint, no cluster. This script does exactly that, widening the locked
reporting set (R@1 / R@10 / MRR / nDCG@10) with R@5, R@100, MAP (already computed but unreported),
and mean / median first-hit rank.

It reuses the SAME pure functions the harness used (`prooflens.eval.metrics`), so the recomputed
R@1 / R@10 / MRR / nDCG@10 / MAP must reproduce each file's stored aggregate exactly — that identity
is asserted as a built-in correctness check (`--check`), and a mismatch exits non-zero.

    python scripts/expand_metrics.py                          # scan results/metrics, print MD table
    python scripts/expand_metrics.py --metrics-dir results/metrics --out results/extended_metrics.md
    python scripts/expand_metrics.py --check                  # only verify recompute == stored

Read-only over existing JSONs; writes nothing unless --out is given.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from prooflens.eval.metrics import (
    _dedupe,
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Extended k-set. R@1/R@10 overlap the locked set (for the correctness check); R@5/R@100 are new.
DEFAULT_KS = [1, 5, 10, 100]
NDCG_K = 10  # matches eval/evaluate.py (docs/EVALUATION.md fixes nDCG@10)


def _first_hit_rank(retrieved: list[str], gold: set[str]) -> int | None:
    """1-indexed rank of the first gold premise in the deduped ranking, or None if absent."""
    for rank, uid in enumerate(_dedupe(retrieved), start=1):
        if uid in gold:
            return rank
    return None


def recompute(records: list[dict], k_list: list[int]) -> dict[str, float | None]:
    """Aggregate extended metrics from per-example records (mean over examples), like _aggregate."""
    n = len(records)
    rk: dict[str, list[float]] = {f"R@{k}": [] for k in k_list}
    mrr: list[float] = []
    ap: list[float] = []
    ndcg: list[float] = []
    first_ranks: list[int] = []
    for r in records:
        ranked = r["retrieved"]
        gold = set(r["gold"])
        for k in k_list:
            rk[f"R@{k}"].append(recall_at_k(ranked, gold, k))
        mrr.append(reciprocal_rank(ranked, gold))
        ap.append(average_precision(ranked, gold))
        ndcg.append(ndcg_at_k(ranked, gold, NDCG_K))
        fr = _first_hit_rank(ranked, gold)
        if fr is not None:
            first_ranks.append(fr)

    out: dict[str, float | None] = {key: (sum(v) / n if n else None) for key, v in rk.items()}
    out["MRR"] = sum(mrr) / n if n else None
    out["MAP"] = sum(ap) / n if n else None
    out[f"nDCG@{NDCG_K}"] = sum(ndcg) / n if n else None
    # First-hit rank is defined only over examples that actually retrieved a gold premise; the
    # coverage (fraction with >=1 hit in the top-retrieve_k) is reported alongside so a low mean
    # rank driven by few hits is visible.
    out["MeanRank"] = statistics.mean(first_ranks) if first_ranks else None
    out["MedianRank"] = statistics.median(first_ranks) if first_ranks else None
    out["AnyHit"] = len(first_ranks) / n if n else None
    return out


def check_against_stored(stored: dict, got: dict, tol: float = 1e-9) -> list[str]:
    """Return mismatch messages for keys present in BOTH the stored aggregate and the recompute."""
    problems = []
    for key, sval in stored.items():
        if key in got and got[key] is not None and sval is not None:
            if abs(got[key] - sval) > tol:
                problems.append(f"{key}: recomputed {got[key]:.6f} != stored {sval:.6f}")
    return problems


def load_eval_json(path: Path) -> dict | None:
    """Load an eval metrics JSON; return None if it isn't one (e.g. summary.csv, malformed)."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(blob, dict) or "provenance" not in blob:
        return None
    return blob


# -- CLI ------------------------------------------------------------------------------------------

COLS = ["R@1", "R@5", "R@10", "R@100", "MRR", "MAP", "nDCG@10", "MeanRank", "MedianRank", "AnyHit"]
PERCENT_COLS = {"R@1", "R@5", "R@10", "R@100", "MRR", "MAP", "nDCG@10", "AnyHit"}


def _fmt(col: str, val: float | None) -> str:
    if val is None:
        return "—"
    if col in PERCENT_COLS:
        return f"{val * 100:.2f}"
    return f"{val:.1f}"  # ranks


def render_markdown(rows: list[dict], k_list: list[int]) -> str:
    """One Markdown table per split, systems as rows. rows: {config, split, n, metrics}."""
    cols = ([f"R@{k}" for k in k_list]
            + ["MRR", "MAP", f"nDCG@{NDCG_K}", "MeanRank", "MedianRank", "AnyHit"])
    lines = ["> Recall/MRR/MAP/nDCG/AnyHit as **percent**; MeanRank/MedianRank = rank of the first "
             "gold premise (over examples with a hit). Recomputed offline from saved records.",
             ""]
    for split in sorted({r["split"] for r in rows}):
        lines.append(f"### {split}")
        lines.append("")
        header = "| system | n | " + " | ".join(cols) + " |"
        sep = "|---|--:|" + "|".join(["--:"] * len(cols)) + "|"
        lines.extend([header, sep])
        for r in sorted((x for x in rows if x["split"] == split), key=lambda x: x["config"]):
            cells = " | ".join(_fmt(c, r["metrics"].get(c)) for c in cols)
            lines.append(f"| {r['config']} | {r['n']} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-dir", default="results/metrics",
                    help="directory of eval *.json (default: results/metrics)")
    ap.add_argument("--files", nargs="*", help="explicit JSON files instead of scanning the dir")
    ap.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS, help="k values for R@k")
    ap.add_argument("--out", help="write the Markdown table to this file (also printed to stdout)")
    ap.add_argument("--check", action="store_true",
                    help="only verify recomputed == stored aggregate; exit non-zero on mismatch")
    args = ap.parse_args()

    paths = ([Path(f) for f in args.files] if args.files
             else sorted(Path(args.metrics_dir).glob("*.json")))
    rows: list[dict] = []
    all_problems: list[str] = []
    skipped: list[str] = []

    for path in paths:
        blob = load_eval_json(path)
        if blob is None:
            continue
        prov = blob.get("provenance", {})
        config = prov.get("config_name", path.stem)
        split = prov.get("split", "?")
        records = blob.get("examples")
        if not records:
            skipped.append(f"{path.name} (no per-example records — cannot recompute)")
            continue
        # Not every results/metrics/*.json is a retrieval eval: the Part-4 generation-comparison
        # files (and any older schema) have `examples[]` without a `retrieved`/`gold` ranking, so
        # no @k metric is defined for them. Skip rather than crash.
        if "retrieved" not in records[0] or "gold" not in records[0]:
            skipped.append(f"{path.name} (records lack retrieved/gold — not a retrieval eval)")
            continue
        got = recompute(records, args.ks)
        problems = check_against_stored(blob.get("metrics", {}), got)
        if problems:
            all_problems.extend(f"{path.name}: {p}" for p in problems)
        rows.append({"config": config, "split": split, "n": len(records), "metrics": got})

    if args.check:
        for s in skipped:
            print(f"  skipped: {s}")
        if all_problems:
            print("CORRECTNESS CHECK FAILED — recompute disagrees with stored aggregate:")
            for p in all_problems:
                print("  " + p)
            raise SystemExit(1)
        print(f"CORRECTNESS CHECK PASSED — recompute == stored for {len(rows)} eval file(s).")
        return

    table = render_markdown(rows, args.ks)
    print(table)
    if skipped:
        print("\n> skipped (no records): " + "; ".join(skipped))
    if all_problems:
        print("\n> ⚠️ CORRECTNESS MISMATCHES (recompute != stored):")
        for p in all_problems:
            print("  " + p)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
