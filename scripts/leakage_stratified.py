"""Decisive test of the cross-split leakage hypothesis (stratified re-scoring).

**The claim under test.** The public ReProver checkpoint is trained on the `random` split. Because
`random` and `novel_premises` are two independent partitions of the SAME theorem pool, most
`novel_premises/test` theorems also sit in `random/train` — so the checkpoint was trained on their
exact (state -> gold premise) pairs, and its novel score (63.66 R@10) is inflated.

**Why the raw overlap statistic is NOT sufficient evidence.** `random/train` holds ~95-97% of all
theorems, so *any* theorem subset overlaps it at ~97%. "97.2% of novel-test theorems are in
random-train" is therefore close to arithmetic, not a smoking gun. It shows the splits share
theorems (true by construction); it does NOT show the overlap *causes* the inflation.

**This script tests causation, and can falsify it.** Partition `novel_premises/test` by theorem
membership in `random/train` and re-score the SAME per-example records in each group:

    LEAKED  (theorem in random/train)     — the model trained on these states
    CLEAN   (theorem NOT in random/train) — a genuine holdout for this checkpoint

  * If leakage drives the number: LEAKED >> CLEAN, and CLEAN should land near the published
    clean-novel reference (~27.6 R@10).
  * If CLEAN ~= LEAKED: the leakage explanation is WRONG and the anomaly has another cause.

The CLEAN group is small (~2.8% of theorems), but the effect under test (63.7 vs 27.6 R@10) is
enormous relative to its standard error, so the test is well powered. A bootstrap CI on the CLEAN
group is reported so the reader can see exactly how much it can be trusted.

Re-scores existing records only — no model, no GPU, no re-run. Reads JSON with a streaming parser
(split files are large), so it is safe on a LOGIN NODE, in seconds:

    DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4 \
        python scripts/leakage_stratified.py \
            --metrics results/metrics/dense_reprover_novel_premises.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from prooflens.data.audit import stream_theorems

METRICS = ["R@1", "R@10", "MRR", "nDCG@10"]
PUBLISHED_CLEAN_NOVEL_R10 = 27.6      # ReProver's own split-matched novel number (LeanDojo T1)


def theorem_ids(split_path: str) -> set[tuple[str, str]]:
    """{(file_path, full_name)} for a split file — theorem identity, as proofs.py uses it."""
    return {(t["file_path"], t["full_name"]) for t in stream_theorems(split_path)}


def full_name_to_paths(split_path: str) -> dict[str, set[str]]:
    """{full_name: {file_path, ...}} — an eid only carries full_name, so we map it back here."""
    out: dict[str, set[str]] = defaultdict(set)
    for t in stream_theorems(split_path):
        out[t["full_name"]].add(t["file_path"])
    return out


def _agg(records: list[dict], metric: str) -> float:
    return float(np.mean([r["metrics"][metric] for r in records])) if records else float("nan")


def _bootstrap_ci(records: list[dict], metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile CI for a group's mean — the CLEAN group is small, so quantify its uncertainty."""
    if not records:
        return (float("nan"), float("nan"))
    vals = np.array([r["metrics"][metric] for r in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def stratify(metrics_json: str, data_root: str, n_boot: int = 10000, seed: int = 42) -> dict:
    blob = json.loads(Path(metrics_json).read_text(encoding="utf-8"))
    prov = blob.get("provenance", {})
    if prov.get("split") != "novel_premises":
        raise ValueError(
            f"this test only applies to the novel_premises split (got {prov.get('split')!r})"
        )

    root = Path(data_root)
    rand_train = theorem_ids(str(root / "random" / "train.json"))
    novel_paths = full_name_to_paths(str(root / "novel_premises" / "test.json"))

    leaked: list[dict] = []
    clean: list[dict] = []
    ambiguous = 0
    unmapped = 0

    for rec in blob["examples"]:
        full_name = rec["eid"].rsplit("#", 1)[0]          # eid = f"{full_name}#{tactic_index}"
        paths = novel_paths.get(full_name)
        if not paths:
            unmapped += 1
            continue
        if len(paths) > 1:
            # same theorem name in several files -> identity is ambiguous from the eid alone.
            # Only safe if ALL candidate paths agree on membership; otherwise drop it.
            memberships = {(p, full_name) in rand_train for p in paths}
            if len(memberships) > 1:
                ambiguous += 1
                continue
            in_train = memberships.pop()
        else:
            in_train = (next(iter(paths)), full_name) in rand_train
        (leaked if in_train else clean).append(rec)

    groups = {"LEAKED (theorem in random/train)": leaked, "CLEAN (never trained on)": clean}
    out: dict = {
        "metrics_json": metrics_json,
        "model": prov.get("model_id"),
        "n_examples_total": len(blob["examples"]),
        "n_leaked": len(leaked),
        "n_clean": len(clean),
        "n_ambiguous_dropped": ambiguous,
        "n_unmapped_dropped": unmapped,
        "groups": {},
    }
    for name, recs in groups.items():
        out["groups"][name] = {
            "n": len(recs),
            **{m: _agg(recs, m) for m in METRICS},
            "R@10_ci95": _bootstrap_ci(recs, "R@10", n_boot, seed),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", required=True,
                    help="an eval metrics JSON for the novel_premises split")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.data_root:
        raise SystemExit("set DATA_ROOT (or pass --data-root)")

    res = stratify(args.metrics, args.data_root, seed=args.seed)

    print(f"\nmodel: {res['model']}")
    print(f"novel_premises/test examples: {res['n_examples_total']}")
    if res["n_ambiguous_dropped"] or res["n_unmapped_dropped"]:
        print(f"dropped: {res['n_ambiguous_dropped']} ambiguous, "
              f"{res['n_unmapped_dropped']} unmapped")
    print()
    print(f"{'group':<34} {'n':>6} {'R@1':>8} {'R@10':>8} {'MRR':>8} {'nDCG@10':>9}   "
          f"{'R@10 95% CI':>18}")
    print("-" * 100)
    for name, g in res["groups"].items():
        lo, hi = g["R@10_ci95"]
        ci = f"[{lo * 100:5.1f},{hi * 100:5.1f}]"
        print(f"{name:<34} {g['n']:>6} {g['R@1'] * 100:>7.2f}% {g['R@10'] * 100:>7.2f}% "
              f"{g['MRR']:>8.4f} {g['nDCG@10']:>9.4f}   {ci:>18}")

    leaked = res["groups"]["LEAKED (theorem in random/train)"]
    clean = res["groups"]["CLEAN (never trained on)"]
    print()
    if clean["n"] == 0:
        print("VERDICT: no clean theorems — this checkpoint cannot be tested this way.")
        return
    delta = (leaked["R@10"] - clean["R@10"]) * 100
    lo, hi = clean["R@10_ci95"]
    print(f"LEAKED - CLEAN  =  {delta:+.2f} pts R@10")
    print(f"published clean-novel reference: {PUBLISHED_CLEAN_NOVEL_R10:.1f} R@10 "
          f"({'INSIDE' if lo * 100 <= PUBLISHED_CLEAN_NOVEL_R10 <= hi * 100 else 'OUTSIDE'} "
          f"the CLEAN group's 95% CI)")
    print()
    if delta > 0 and hi * 100 < leaked["R@10"] * 100:
        print("VERDICT: LEAKAGE CONFIRMED — the model scores far higher on theorems it was")
        print("         trained on. The headline novel number is inflated by memorisation.")
    else:
        print("VERDICT: LEAKAGE NOT SUPPORTED — the clean holdout scores as well as the leaked")
        print("         group. The novel anomaly has some OTHER cause; the explanation in")
        print("         comparison.md must be retracted and re-investigated.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
