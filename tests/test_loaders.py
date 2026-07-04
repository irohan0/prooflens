"""Unit tests for data/proofs.py (the split loader) against the mini fixtures.

Asserts the production loader reproduces exactly the gold sets and usable-example set derived by
hand in tests/fixtures/EXPECTED.md and proven by the Phase 3 smoke oracle — including dropping the
un-locatable `foo_missing` provenance (empty gold -> tactic skipped) and the no-tactic theorem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"

A = "Mathlib/Algebra/Basic.lean"
B = "Mathlib/Order/Basic.lean"
C = "Mathlib/Topology/Basic.lean"

# expected UIDs (path::full_name@line,col; all fixture premises start at column 1)
ADD = f"{A}::add_comm@10,1"
LEREFL = f"{B}::le_refl@5,1"
LETRANS = f"{B}::le_trans@15,1"
CONT = f"{C}::continuous_id@25,1"


def _examples():
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    return corpus, list(load_split(str(FIX), "random", corpus))


def test_usable_examples_match_contract():
    _, examples = _examples()
    by_id = {e.eid: e for e in examples}
    # exactly 3 usable examples; foo tactic (unlocatable) and no-tactic theorem excluded
    assert set(by_id) == {
        "continuous_const_fixture#0",
        "continuous_const_fixture#1",
        "le_of_lt_fixture#0",
    }


def test_gold_sets_match_contract():
    _, examples = _examples()
    by_id = {e.eid: e for e in examples}
    assert by_id["continuous_const_fixture#0"].gold == frozenset({CONT})
    assert by_id["continuous_const_fixture#1"].gold == frozenset({LEREFL})
    assert by_id["le_of_lt_fixture#0"].gold == frozenset({LETRANS, ADD})


def test_example_carries_accessibility_context():
    _, examples = _examples()
    by_id = {e.eid: e for e in examples}
    e1 = by_id["continuous_const_fixture#0"]
    assert e1.file_path == C and e1.thm_pos == (40, 1)
    assert e1.state == "⊢ continuous_id applied here"
    e2 = by_id["le_of_lt_fixture#0"]
    assert e2.file_path == B and e2.thm_pos == (30, 1)


def test_gold_subset_of_accessible_invariant():
    # the Phase 2 invariant: every gold premise is accessible to its theorem
    corpus, examples = _examples()
    for e in examples:
        acc = accessible_premises(corpus, e.file_path, e.thm_pos)
        assert set(e.gold) <= acc, f"gold not accessible in {e.eid}"


def test_unknown_split_rejected():
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    with pytest.raises(ValueError):
        list(load_split(str(FIX), "not_a_split", corpus))
