"""Pure, fully unit-tested retrieval metrics: Recall@k, MRR, nDCG@k.

These are the scoring floor for the whole project (build order step 1): if the metrics are
wrong, every downstream number is wrong. Definitions follow docs/EVALUATION.md §2.

Each function scores ONE evaluation example (one tactic's proof state). Averaging over examples
is the aggregator's job (eval/evaluate.py), never done here — the diagnostics in
docs/EVALUATION.md §3 call out "averaging over a flattened premise list" as a classic bug, so we
keep per-example scoring pure and separate.

Conventions (documented decisions for the edge cases EVALUATION.md §2 asks us to pin down):
- `ranked` is best-first and may contain duplicates; we **dedupe keeping the best (first) rank**
  before scoring, so a name that appears twice never inflates a count nor occupies two positions.
- **Empty gold is not a valid evaluation example** — the protocol only uses tactics with >=1
  premise. Rather than silently returning 0/NaN and hiding an upstream bug, every metric raises
  ValueError on empty gold so a mistake in split loading fails loudly.
- Ranks are 1-indexed. `k` must be >= 1.
"""

from __future__ import annotations

import math


def _dedupe(ranked: list[str]) -> list[str]:
    """Return `ranked` with duplicates removed, keeping each name at its first (best) rank."""
    seen: set[str] = set()
    out: list[str] = []
    for name in ranked:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _validate(gold: set[str], k: int | None = None) -> None:
    if not gold:
        raise ValueError(
            "empty gold set: not a valid evaluation example "
            "(the protocol excludes tactics with no premises — check split loading)"
        )
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Recall@k = |{top-k of ranked} ∩ gold| / |gold| for a single example.

    With |gold| > 1, recall@k is naturally capped below 1 when k < |gold| (you cannot retrieve
    more distinct gold items than there are positions). That is the intended ReProver definition.
    """
    _validate(gold, k)
    topk = _dedupe(ranked)[:k]
    hits = sum(1 for name in topk if name in gold)
    return hits / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1 / (rank of the first gold premise in `ranked`); 0.0 if no gold premise is present."""
    _validate(gold)
    for rank, name in enumerate(_dedupe(ranked), start=1):
        if name in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """nDCG@k with binary relevance (gold = 1, else 0): DCG@k / IDCG@k for a single example.

    DCG@k  = sum over top-k positions i of rel_i / log2(i + 1).
    IDCG@k = the same with all min(k, |gold|) gold items ranked first (the ideal ordering).
    Returns 0.0 when no gold premise falls in the top-k (DCG = 0).
    """
    _validate(gold, k)
    topk = _dedupe(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, name in enumerate(topk, start=1) if name in gold)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: list[str], gold: set[str], k: int | None = None) -> float:
    """Average Precision for one example (mean of AP over examples = MAP).

    AP = (1 / |gold|) * sum over gold hits of (precision at the rank of that hit). This is a
    standard retrieval metric teammates may expect; it is NOT part of the locked ReProver
    reporting set (R@1, R@10, MRR, nDCG@10), so it is an optional extra column.

    `k` optionally truncates the ranking (AP@k); default None uses the full list. Because we
    persist the top-100 per example, AP@k for any k<=100 can be recomputed post-hoc without
    re-running retrieval — the same is true of every @k metric here.
    """
    _validate(gold)
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    ranking = _dedupe(ranked)
    if k is not None:
        ranking = ranking[:k]
    hits = 0
    precision_sum = 0.0
    for rank, name in enumerate(ranking, start=1):
        if name in gold:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(gold)
