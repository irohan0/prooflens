"""Structural-vs-lexical stratification — does late interaction's novel-premise advantage survive
when the gold name is NOT in the proof state? (The "neural BM25" refutation.)

**The objection this answers.** Late interaction (LI) beats the single-vector control on
`novel_premises`. A sceptic can say: novel premises are more lexically distinctive (the gold name
appears literally in the state more often — 26.6% novel vs 19.4% random, from the Phase-11 audit),
and LI is more lexically sensitive than mean-pooled cosine, so maybe LI is just a soft neural BM25
winning on surface overlap — not on any deeper *structural* matching.

**The test.** Split every novel example into two buckets by the SAME rule the audit uses
(`short_name(gold) in lean_tokenize(state)`):
  - LEXICAL    — a gold premise's short name is a token of the proof state (surface overlap present)
  - STRUCTURAL — it is NOT (retrieval must rely on structure/semantics, not name matching)
Then compare LI vs the matched single-vector control (SV) WITHIN each bucket, example-paired.

**Interpretation (pre-registered).**
  - If LI's advantage over SV persists (ideally grows) in the STRUCTURAL bucket → the advantage is
    NOT explained by lexical overlap; it is structural (thesis is deep, not a neural-BM25 effect).
  - If LI only wins in the LEXICAL bucket and ties/loses in STRUCTURAL → the advantage IS largely
    lexical; report that honestly (it weakens the mechanistic claim but is the truth).

Also surfaces the top STRUCTURAL wins — novel examples where LI ranks a gold premise in the top-10
and SV does not, with the gold names and each side's best rank — as qualitative evidence for the
write-up.

Eval-only: joins the two runs' per-example records (from eval/evaluate.py) with the proof states
(from the split loader). No model, no GPU — safe on a LOGIN NODE (needs the benchmark + the two
metrics JSONs). Reuses the audit's gold-name-in-state rule so the buckets match its 26.6% stat.

    DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4 \
        python scripts/lexical_stratified.py \
            --li results/metrics/late_interaction_ft_novel_weighted_novel_premises.json \
            --sv results/metrics/dense_sv_ft_novel_lr3e6_novel_premises.json \
            --split novel_premises
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from prooflens.data.audit import short_name, uid_full_name
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split
from prooflens.retrievers.bm25 import lean_tokenize

METRICS = ["R@1", "R@10"]


def load_records(path: str) -> tuple[dict, dict[str, dict]]:
    """Return (provenance, {eid: record}) from an eval metrics JSON."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return blob.get("provenance", {}), {r["eid"]: r for r in blob["examples"]}


def gold_name_in_state(state: str, gold_uids) -> bool:
    """The audit's exact rule: a gold premise's short name is a token of the proof state."""
    toks = set(lean_tokenize(state))
    return any(short_name(uid_full_name(u)) in toks for u in gold_uids)


def build_eid_context(corpus_path: str, splits_dir: str, split: str) -> dict[str, dict]:
    """{eid: {lexical: bool, state: str, gold_names: [short...]}} for every example in the split."""
    corpus = load_corpus(corpus_path)
    out: dict[str, dict] = {}
    for ex in load_split(splits_dir, split, corpus):
        out[ex.eid] = {
            "lexical": gold_name_in_state(ex.state, ex.gold),
            "gold_names": sorted({short_name(uid_full_name(u)) for u in ex.gold}),
        }
    return out


def _mean(recs: list[dict], metric: str) -> float:
    return float(np.mean([r["metrics"][metric] for r in recs])) if recs else float("nan")


def _paired_delta_ci(li: list[dict], sv: list[dict], metric: str, n_boot: int,
                     rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean paired (LI-SV) per-example difference for a bucket, with a percentile bootstrap CI."""
    d = np.array([a["metrics"][metric] - b["metrics"][metric]
                  for a, b in zip(li, sv, strict=True)], dtype=np.float64)
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))


def stratify(li_path: str, sv_path: str, corpus_path: str, splits_dir: str, split: str,
             n_boot: int = 10000, seed: int = 42, top_examples: int = 15) -> dict:
    prov_li, li = load_records(li_path)
    prov_sv, sv = load_records(sv_path)
    for name, p in (("LI", prov_li), ("SV", prov_sv)):
        if p.get("split") and p["split"] != split:
            raise ValueError(f"{name} run is split {p['split']!r}, expected {split!r}")
    ctx = build_eid_context(corpus_path, splits_dir, split)

    eids = [e for e in li if e in sv and e in ctx]           # examples all three agree on
    buckets: dict[str, dict[str, list[dict]]] = {
        "LEXICAL (gold name in state)": {"li": [], "sv": []},
        "STRUCTURAL (gold name NOT in state)": {"li": [], "sv": []},
    }
    for e in eids:
        key = ("LEXICAL (gold name in state)" if ctx[e]["lexical"]
               else "STRUCTURAL (gold name NOT in state)")
        buckets[key]["li"].append(li[e])
        buckets[key]["sv"].append(sv[e])

    rng = np.random.default_rng(seed)
    out: dict = {
        "li_run": Path(li_path).name, "sv_run": Path(sv_path).name, "split": split,
        "n_examples": len(eids), "buckets": {},
    }
    for name, b in buckets.items():
        row = {"n": len(b["li"])}
        for m in METRICS:
            li_m, sv_m = _mean(b["li"], m), _mean(b["sv"], m)
            delta, lo, hi = _paired_delta_ci(b["li"], b["sv"], m, n_boot, rng)
            row[m] = {"li": li_m, "sv": sv_m, "delta": delta, "ci": [lo, hi],
                      "significant": bool(lo > 0 or hi < 0)}
        out["buckets"][name] = row

    # qualitative: structural examples where LI got a gold hit in top-10 and SV did not
    wins = []
    for e in eids:
        if ctx[e]["lexical"]:
            continue
        li_hit = li[e]["metrics"]["R@10"] > 0
        sv_hit = sv[e]["metrics"]["R@10"] > 0
        if li_hit and not sv_hit:
            def best_rank(rec):
                rs = [r for r in rec["hit_ranks"].values() if r is not None]
                return min(rs) if rs else None
            wins.append({"eid": e, "gold_names": ctx[e]["gold_names"],
                         "li_best_rank": best_rank(li[e]), "sv_best_rank": best_rank(sv[e])})
    wins.sort(key=lambda w: (w["li_best_rank"] or 999))
    out["structural_li_wins"] = {"n_total": len(wins), "examples": wins[:top_examples]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--li", required=True, help="late-interaction per-example metrics JSON")
    ap.add_argument("--sv", required=True, help="single-vector control per-example metrics JSON")
    ap.add_argument("--split", default="novel_premises")
    ap.add_argument("--corpus", default=None, help="corpus.jsonl (default $DATA_ROOT/corpus.jsonl)")
    ap.add_argument("--splits-dir", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    splits_dir = args.splits_dir or os.environ.get("DATA_ROOT")
    if not splits_dir:
        raise SystemExit("set DATA_ROOT (or pass --splits-dir)")
    corpus = args.corpus or str(Path(splits_dir) / "corpus.jsonl")

    res = stratify(args.li, args.sv, corpus, splits_dir, args.split, seed=args.seed)

    print(f"\nLI: {res['li_run']}\nSV: {res['sv_run']}\nsplit: {res['split']}   "
          f"paired examples: {res['n_examples']}\n")
    print(f"{'bucket':<38}{'n':>6}{'metric':>8}{'LI':>8}{'SV':>8}{'LI-SV':>9}  {'95% CI':>16}")
    print("-" * 96)
    for name, b in res["buckets"].items():
        for i, m in enumerate(METRICS):
            c = b[m]
            ci = f"[{c['ci'][0] * 100:+.1f},{c['ci'][1] * 100:+.1f}]"
            star = " *" if c["significant"] else ""
            label = f"{name}" if i == 0 else ""
            nlab = f"{b['n']}" if i == 0 else ""
            print(f"{label:<38}{nlab:>6}{m:>8}{c['li'] * 100:>7.1f}%{c['sv'] * 100:>7.1f}%"
                  f"{c['delta'] * 100:>+8.1f}{ci:>17}{star}")
    print("\n(* = paired bootstrap CI on LI-SV excludes 0. Positive LI-SV = LI ahead.)")

    struct = res["buckets"]["STRUCTURAL (gold name NOT in state)"]["R@10"]
    verdict = ("STRUCTURAL — LI's novel advantage survives without lexical overlap: NOT neural BM25"
               if struct["delta"] > 0 and struct["ci"][0] > 0
               else "structural-bucket LI-SV gap is not clearly positive — advantage largely "
                    "lexical; report honestly")
    print(f"\nVERDICT: {verdict}")
    w = res["structural_li_wins"]
    print(f"\nStructural LI wins (top-10 hit, SV missed): {w['n_total']} examples. Sample:")
    for ex in w["examples"][:8]:
        print(f"  gold={ex['gold_names']}  LI rank {ex['li_best_rank']}  SV {ex['sv_best_rank']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
