"""Tiny end-to-end smoke test — no downloads, no cluster (build order step 2).

Proves the *shape* of the harness on hand-made fixtures that mirror the real LeanDojo Benchmark 4
schema (confirmed in Phase 2): raw corpus + split files -> evaluation examples (state, gold,
accessible) -> a retriever restricted to the accessible set -> the real metrics -> hand-known
numbers (see tests/fixtures/EXPECTED.md).

The `_reference` helpers below are a TEST ORACLE, deliberately not the production loaders. Phase 4
implements the production data loaders in `prooflens/data/` and its tests assert they reproduce
exactly these gold/accessible sets on the same fixtures. Keeping the oracle here lets the smoke
test derive everything from the fixture files (catching typos) while honouring the build order.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

from prooflens.eval.metrics import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from prooflens.retrievers.base import Retriever

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


def uid(path: str, full_name: str, start) -> str:
    # collision-free id matching production Premise.uid (see data/corpus.py)
    return f"{path}::{full_name}@{start[0]},{start[1]}"


# --------------------------------------------------------------------------------------------
# Test oracle: raw fixtures -> (premises, import graph) and (state, gold, accessible) examples.
# Mirrors ReProver's locate_premise (position-containment) and accessibility (transitive imports
# + earlier-in-same-file). Validated against the real data in Phase 2.
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    eid: str
    state: str
    gold: frozenset[str]
    accessible: frozenset[str]


def _load_corpus():
    """Return (order, by_path, imports) where order is [(uid, full_name)] in file order."""
    order: list[tuple[str, str]] = []
    by_path: dict[str, list[dict]] = {}
    imports: dict[str, list[str]] = {}
    with open(FIX / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_path[rec["path"]] = rec["premises"]
            imports[rec["path"]] = rec["imports"]
            for p in rec["premises"]:
                order.append((uid(rec["path"], p["full_name"], p["start"]), p["full_name"]))
    return order, by_path, imports


def _transitive_imports(path: str, imports: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    q = deque(imports.get(path, []))
    while q:
        f = q.popleft()
        if f in seen:
            continue
        seen.add(f)
        q.extend(imports.get(f, []))
    return seen


def _locate_gold(prov: dict, by_path: dict[str, list[dict]]) -> str | None:
    path = prov["def_path"]
    pos = tuple(prov["def_pos"])
    for p in by_path.get(path, []):
        if tuple(p["start"]) <= pos <= tuple(p["end"]):
            return uid(path, p["full_name"], p["start"])
    return None


def _accessible_uids(thm: dict, by_path, imports) -> set[str]:
    acc: set[str] = set()
    for f in _transitive_imports(thm["file_path"], imports):     # imported files: all premises
        for p in by_path.get(f, []):
            acc.add(uid(f, p["full_name"], p["start"]))
    thm_start = tuple(thm["start"])                              # same file: defined earlier
    for p in by_path.get(thm["file_path"], []):
        if tuple(p["end"]) <= thm_start:
            acc.add(uid(thm["file_path"], p["full_name"], p["start"]))
    return acc


def _build_examples() -> list[Example]:
    _, by_path, imports = _load_corpus()
    data = json.loads((FIX / "random" / "test.json").read_text(encoding="utf-8"))
    examples: list[Example] = []
    for t in data:
        acc = _accessible_uids(t, by_path, imports)
        for j, tac in enumerate(t.get("traced_tactics", [])):
            _, provs = tac["annotated_tactic"]
            gold = {g for g in (_locate_gold(p, by_path) for p in provs) if g is not None}
            if not gold:                                        # drop tactics with no located gold
                continue
            examples.append(
                Example(f'{t["full_name"]}#{j}', tac["state_before"],
                        frozenset(gold), frozenset(acc))
            )
    return examples


# --------------------------------------------------------------------------------------------
# Trivial retriever (test-only): deterministic substring ranker over the accessible set.
# --------------------------------------------------------------------------------------------

class SubstringRetriever(Retriever):
    """Rank accessible premises: score 1 if the premise base-name is a substring of the state,
    else 0; tie-break by corpus order. Deterministic, so metrics are hand-computable."""

    def build_index(self, corpus) -> None:
        # corpus: list[(uid, full_name)] in file order
        self.items = [(u, fn.split(".")[-1]) for u, fn in corpus]

    def retrieve(self, state: str, accessible: set[str], k: int) -> list[tuple[str, float]]:
        scored = [
            (u, 1.0 if base in state else 0.0, i)
            for i, (u, base) in enumerate(self.items)
            if u in accessible
        ]
        scored.sort(key=lambda t: (-t[1], t[2]))
        return [(u, s) for u, s, _ in scored[:k]]


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------

def test_fixture_integrity():
    order, by_path, imports = _load_corpus()
    assert len(order) == 6
    # every premise has the real schema fields and well-formed positions
    for prems in by_path.values():
        for p in prems:
            assert set(p) >= {"full_name", "code", "start", "end", "kind"}
            assert len(p["start"]) == 2 and len(p["end"]) == 2
            assert tuple(p["start"]) <= tuple(p["end"])
    # import targets resolve to corpus paths (self-contained graph)
    targets = {t for ts in imports.values() for t in ts}
    assert targets <= set(by_path)


def test_examples_match_expected():
    examples = {e.eid: e for e in _build_examples()}
    # exactly 3 usable examples; the `foo` tactic and the no-tactic theorem contribute none
    assert len(examples) == 3
    assert set(examples) == {
        "continuous_const_fixture#0",
        "continuous_const_fixture#1",
        "le_of_lt_fixture#0",
    }
    assert examples["continuous_const_fixture#0"].gold == {
        "Mathlib/Topology/Basic.lean::continuous_id@25,1"
    }
    assert examples["continuous_const_fixture#1"].gold == {
        "Mathlib/Order/Basic.lean::le_refl@5,1"
    }
    assert examples["le_of_lt_fixture#0"].gold == {
        "Mathlib/Order/Basic.lean::le_trans@15,1",
        "Mathlib/Algebra/Basic.lean::add_comm@10,1",
    }


def test_accessibility_filter():
    examples = {e.eid: e for e in _build_examples()}
    # invariant validated in Phase 2: gold is always accessible
    for e in examples.values():
        assert e.gold <= e.accessible, f"gold not accessible in {e.eid}"
    # theorem 1 (Topology) sees all 6; theorem 2 (Order) excludes the Topology premises
    assert len(examples["continuous_const_fixture#0"].accessible) == 6
    t2 = examples["le_of_lt_fixture#0"].accessible
    assert t2 == {
        "Mathlib/Algebra/Basic.lean::add_comm@10,1",
        "Mathlib/Algebra/Basic.lean::mul_comm@20,1",
        "Mathlib/Order/Basic.lean::le_refl@5,1",
        "Mathlib/Order/Basic.lean::le_trans@15,1",
    }
    assert "Mathlib/Topology/Basic.lean::continuous_id@25,1" not in t2


def _fit_retriever() -> SubstringRetriever:
    order, _, _ = _load_corpus()
    r = SubstringRetriever()
    r.build_index(order)
    return r


def test_retriever_wellformed():
    r = _fit_retriever()
    for e in _build_examples():
        res = r.retrieve(e.state, set(e.accessible), k=100)
        uids = [u for u, _ in res]
        scores = [s for _, s in res]
        assert len(uids) == len(set(uids)), "duplicate UIDs returned"
        assert set(uids) <= set(e.accessible), "returned a non-accessible premise"
        assert scores == sorted(scores, reverse=True), "scores not best-first"
    # k truncation
    e = next(x for x in _build_examples() if x.eid == "le_of_lt_fixture#0")
    assert len(r.retrieve(e.state, set(e.accessible), k=1)) == 1


def test_end_to_end_metrics_per_example():
    r = _fit_retriever()
    ex = {e.eid: e for e in _build_examples()}

    def ranked(eid: str) -> list[str]:
        e = ex[eid]
        return [u for u, _ in r.retrieve(e.state, set(e.accessible), k=100)]

    # 1a and 1b: gold at rank 1 -> everything perfect
    for eid in ("continuous_const_fixture#0", "continuous_const_fixture#1"):
        g = set(ex[eid].gold)
        rk = ranked(eid)
        assert recall_at_k(rk, g, 1) == pytest.approx(1.0)
        assert recall_at_k(rk, g, 10) == pytest.approx(1.0)
        assert reciprocal_rank(rk, g) == pytest.approx(1.0)
        assert ndcg_at_k(rk, g, 10) == pytest.approx(1.0)
        assert average_precision(rk, g) == pytest.approx(1.0)

    # 2a: two gold, ranked 1 and 2 -> R@1 = 0.5, the rest 1.0
    g = set(ex["le_of_lt_fixture#0"].gold)
    rk = ranked("le_of_lt_fixture#0")
    assert rk[:2] == [
        "Mathlib/Order/Basic.lean::le_trans@15,1",
        "Mathlib/Algebra/Basic.lean::add_comm@10,1",
    ]
    assert recall_at_k(rk, g, 1) == pytest.approx(0.5)
    assert recall_at_k(rk, g, 10) == pytest.approx(1.0)
    assert reciprocal_rank(rk, g) == pytest.approx(1.0)
    assert ndcg_at_k(rk, g, 10) == pytest.approx(1.0)
    assert average_precision(rk, g) == pytest.approx(1.0)


def test_end_to_end_metrics_aggregate():
    r = _fit_retriever()
    examples = _build_examples()
    assert len(examples) == 3

    agg = {"r1": [], "r10": [], "rr": [], "ndcg10": [], "ap": []}
    for e in examples:
        rk = [u for u, _ in r.retrieve(e.state, set(e.accessible), k=100)]
        g = set(e.gold)
        agg["r1"].append(recall_at_k(rk, g, 1))
        agg["r10"].append(recall_at_k(rk, g, 10))
        agg["rr"].append(reciprocal_rank(rk, g))
        agg["ndcg10"].append(ndcg_at_k(rk, g, 10))
        agg["ap"].append(average_precision(rk, g))

    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    assert mean["r1"] == pytest.approx(2.5 / 3)      # (1 + 1 + 0.5) / 3
    assert mean["r10"] == pytest.approx(1.0)
    assert mean["rr"] == pytest.approx(1.0)           # MRR
    assert mean["ndcg10"] == pytest.approx(1.0)
    assert mean["ap"] == pytest.approx(1.0)           # MAP
