"""Hermetic tests for data/audit.py (Phase 11 pre-fine-tuning audit — model-free stats).

Expected values are hand-derived from the mini fixtures (tests/fixtures/mini_benchmark), whose
three usable examples are:
  * continuous_const_fixture#0  state "⊢ continuous_id applied here"  gold {continuous_id}
  * continuous_const_fixture#1  state "x : α ⊢ le_refl x"            gold {le_refl}
  * le_of_lt_fixture#0          state "a b : α ⊢ le_trans of chain"  gold {le_trans, add_comm}
(the foo_missing tactic is dropped — un-locatable gold — and the no-tactic theorem contributes none,
exactly as load_split does). No torch/pylate here; runs anywhere.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from prooflens.data.audit import (
    _stats,
    _stream_json_array,
    audit_examples,
    frequency_summary,
    short_name,
    stream_examples,
    uid_full_name,
)
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import Example, load_split

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


def _corpus():
    return load_corpus(str(FIX / "corpus.jsonl"))


def _examples():
    return list(load_split(str(FIX), "random", _corpus(), "test.json"))


# -- small pure helpers ---------------------------------------------------------------------------

def test_uid_full_name_and_short_name():
    assert uid_full_name("Mathlib/Algebra/Basic.lean::Nat.add_comm@10,1") == "Nat.add_comm"
    assert short_name("Nat.add_comm") == "add_comm"
    assert short_name("add_comm") == "add_comm"           # no dotted prefix


def test_stats_empty_and_values():
    assert _stats([]) == {
        "n": 0, "mean": None, "median": None, "min": None, "p90": None, "max": None,
    }
    s = _stats([1, 1, 2])
    assert s["n"] == 3 and s["min"] == 1 and s["max"] == 2
    assert abs(s["mean"] - 4 / 3) < 1e-9 and s["median"] == 1


def test_frequency_summary_handcomputed():
    # four distinct premises, one gold-occurrence each
    freq = Counter({"P::a@1,1": 1, "P::b@1,1": 1, "P::c@1,1": 1, "P::d@1,1": 1})
    fs = frequency_summary(freq)
    assert fs["n_unique_gold_premises"] == 4
    assert fs["n_positive_pairs"] == 4
    assert fs["max_count"] == 1
    assert fs["singleton_fraction"] == 1.0
    assert fs["head_coverage_top1pct"] == 0.25          # top-1 of 4 = 1/4 of pairs
    assert sorted(n for n, _ in fs["top_premises"]) == ["a", "b", "c", "d"]

    # a skewed distribution: one head premise dominates
    skew = Counter({"P::head@1,1": 90, "P::x@1,1": 5, "P::y@1,1": 3, "P::z@1,1": 2})
    fs2 = frequency_summary(skew)
    assert fs2["max_count"] == 90
    assert abs(fs2["head_coverage_top1pct"] - 0.90) < 1e-9   # top-1 = 90/100


# -- audit_examples on the fixture ----------------------------------------------------------------

def test_audit_examples_without_corpus():
    a = audit_examples(_examples())            # no accessibility (corpus omitted)
    assert a["n_examples"] == 3
    assert a["n_positive_pairs"] == 4          # |gold| = 1 + 1 + 2
    assert a["gold_size"]["max"] == 2 and a["gold_size"]["min"] == 1
    assert abs(a["gold_size"]["mean"] - 4 / 3) < 1e-9
    assert a["gold_name_in_state_rate"] == 1.0  # every fixture state literally names its gold
    assert a["accessible_size"]["n"] == 0       # not computed without a corpus
    assert a["gold_in_accessible_rate"] is None
    assert a["premise_frequency"]["n_unique_gold_premises"] == 4


def test_audit_examples_with_corpus_accessibility():
    a = audit_examples(_examples(), _corpus())
    # accessible sets: the two continuous_const tactics share the Topology theorem (6 accessible),
    # le_of_lt sees Order + its Algebra import (4 accessible).
    assert sorted([a2 for a2 in _acc_sizes()]) == [4, 6, 6]
    assert a["accessible_size"]["max"] == 6 and a["accessible_size"]["min"] == 4
    assert a["gold_in_accessible_rate"] == 1.0   # 100% of located gold is accessible (Phase 4)


def _acc_sizes():
    from prooflens.data.accessibility import accessible_premises
    corpus = _corpus()
    out = []
    seen = {}
    for ex in _examples():
        key = (ex.file_path, ex.thm_pos)
        if key not in seen:
            seen[key] = len(accessible_premises(corpus, ex.file_path, ex.thm_pos))
        out.append(seen[key])
    return out


def test_gold_name_in_state_can_be_false():
    # a synthetic example whose state does NOT name its gold premise -> rate 0
    ex = Example(eid="e", theorem="t", file_path="f", thm_pos=(1, 1),
                 state="⊢ some goal without any lemma name", gold=frozenset({"F::my_lemma@1,1"}))
    a = audit_examples([ex])
    assert a["gold_name_in_state_rate"] == 0.0


# -- streaming reader -----------------------------------------------------------------------------

def test_stream_examples_matches_load_split():
    corpus = _corpus()
    streamed = list(stream_examples(str(FIX / "random" / "test.json"), corpus))
    loaded = list(load_split(str(FIX), "random", corpus, "test.json"))
    assert [e.eid for e in streamed] == [e.eid for e in loaded]
    assert [sorted(e.gold) for e in streamed] == [sorted(e.gold) for e in loaded]


def test_stream_examples_max_theorems_bounds_the_pass():
    corpus = _corpus()
    # only the FIRST theorem (continuous_const_fixture) -> its two tactics
    streamed = list(stream_examples(str(FIX / "random" / "test.json"), corpus, max_theorems=1))
    assert [e.eid for e in streamed] == ["continuous_const_fixture#0", "continuous_const_fixture#1"]


def test_stream_json_array_is_robust_to_braces_in_strings(tmp_path):
    # a top-level array whose string values contain braces/quotes/commas -> must NOT confuse the
    # decoder (this is why we use json.raw_decode, not brace counting; Lean `code` has literal {}).
    payload = [
        {"full_name": "a", "code": "by { intro x, exact ⟨1, 2⟩ }"},
        {"full_name": "b", "code": "fun s => s ++ \"}],[{\""},
    ]
    p = tmp_path / "arr.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    got = list(_stream_json_array(str(p)))
    assert got == payload
    assert list(_stream_json_array(str(p), max_items=1)) == payload[:1]
