"""Hermetic tests for data/pairs.py (Phase 12 training-pair construction).

Uses the mini fixtures (tests/fixtures/mini_benchmark). The three usable examples give four unique
(state, positive) pairs:
  * "⊢ continuous_id applied here"  -> continuous_id     (theorem gold {continuous_id, le_refl})
  * "x : α ⊢ le_refl x"             -> le_refl           (theorem gold {continuous_id, le_refl})
  * "a b : α ⊢ le_trans of chain"   -> le_trans          (theorem gold {le_trans, add_comm})
  * "a b : α ⊢ le_trans of chain"   -> add_comm          (theorem gold {le_trans, add_comm})
The `random` negative strategy needs no bm25s; the BM25 path's band logic is unit-tested pure and
its bm25s adapter is smoke-tested behind `importorskip`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from prooflens.data.corpus import load_corpus
from prooflens.data.pairs import (
    BM25Miner,
    RandomMiner,
    build_triplets,
    make_accessible_check,
    select_negatives_from_ranking,
)
from prooflens.retrievers.bm25 import premise_document

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"

CONT = "Mathlib/Topology/Basic.lean::continuous_id@25,1"
LEREFL = "Mathlib/Order/Basic.lean::le_refl@5,1"
LETRANS = "Mathlib/Order/Basic.lean::le_trans@15,1"
ADD = "Mathlib/Algebra/Basic.lean::add_comm@10,1"
MUL = "Mathlib/Algebra/Basic.lean::mul_comm@20,1"
ISOPEN = "Mathlib/Topology/Basic.lean::isOpen_univ@8,1"

# per-state theorem-gold (the de-noising exclude set) and the resulting allowed negatives
STATE_TGOLD = {
    "⊢ continuous_id applied here": {CONT, LEREFL},
    "x : α ⊢ le_refl x": {CONT, LEREFL},
    "a b : α ⊢ le_trans of chain": {LETRANS, ADD},
}
STATE_ALLOWED_NEG = {                       # accessible \ theorem_gold, per state
    "⊢ continuous_id applied here": {ISOPEN, LETRANS, ADD, MUL},
    "x : α ⊢ le_refl x": {ISOPEN, LETRANS, ADD, MUL},
    "a b : α ⊢ le_trans of chain": {LEREFL, MUL},
}


def _corpus():
    return load_corpus(str(FIX / "corpus.jsonl"))


def _text_to_uid(corpus):
    return {premise_document(p.full_name, p.code): p.uid for p in corpus.all_premises}


def _fixture_theorems():
    return json.loads((FIX / "random" / "test.json").read_text(encoding="utf-8"))


def _write_train(tmp_path, theorems) -> str:
    d = tmp_path / "random"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "train.json"                     # name matters: the builder keys off it
    p.write_text(json.dumps(theorems), encoding="utf-8")
    return str(p)


# -- integrity: never build from test.json --------------------------------------------------------

def test_refuses_test_json():
    corpus = _corpus()
    with pytest.raises(ValueError, match="test data must never enter training"):
        list(build_triplets(corpus, str(FIX / "random" / "test.json"), negatives="random"))


# -- end-to-end on the fixture (random negatives) -------------------------------------------------

def test_build_random_negatives_integrity_and_determinism(tmp_path):
    corpus = _corpus()
    path = _write_train(tmp_path, _fixture_theorems())
    t2u = _text_to_uid(corpus)

    rows = list(build_triplets(corpus, path, negatives="random", n_neg=2, seed=7))
    assert len(rows) == 4                    # four unique (state, positive) pairs

    for r in rows:
        assert r["query"] in STATE_TGOLD
        pos_uid = t2u[r["positive"]]
        neg_uids = [t2u[t] for t in r["negatives"]]
        assert pos_uid not in neg_uids
        # de-noising: no negative is gold anywhere in the same theorem
        assert all(u not in STATE_TGOLD[r["query"]] for u in neg_uids)
        # negatives are accessible & non-gold for that state
        assert all(u in STATE_ALLOWED_NEG[r["query"]] for u in neg_uids)

    # positives cover exactly the four gold premises
    assert {t2u[r["positive"]] for r in rows} == {CONT, LEREFL, LETRANS, ADD}
    # determinism under a fixed seed
    assert list(build_triplets(corpus, path, negatives="random", n_neg=2, seed=7)) == rows


def test_le_of_lt_negatives_are_the_only_two_allowed(tmp_path):
    # state "a b : α ⊢ le_trans of chain" has exactly two allowed negatives {le_refl, mul_comm}
    corpus = _corpus()
    path = _write_train(tmp_path, _fixture_theorems())
    t2u = _text_to_uid(corpus)
    rows = [r for r in build_triplets(corpus, path, negatives="random", n_neg=2, seed=1)
            if r["query"] == "a b : α ⊢ le_trans of chain"]
    for r in rows:
        assert {t2u[t] for t in r["negatives"]} <= {LEREFL, MUL}


# -- dedup ----------------------------------------------------------------------------------------

def test_dedup_collapses_identical_pairs(tmp_path):
    corpus = _corpus()
    theorems = _fixture_theorems()
    path = _write_train(tmp_path, theorems + theorems)     # every theorem duplicated
    rows = list(build_triplets(corpus, path, negatives="random", n_neg=1, seed=3))
    assert len(rows) == 4                                   # still four, not eight


# -- head-capping ---------------------------------------------------------------------------------

def _repeated_positive_theorems(n: int) -> list[dict]:
    # n theorems, each a single tactic whose gold is add_comm (accessible from Order via import),
    # with DISTINCT states -> freq[add_comm] = n, no dedup.
    out = []
    for i in range(n):
        out.append({
            "file_path": "Mathlib/Order/Basic.lean",
            "full_name": f"thm_{i}",
            "start": [30, 1],
            "end": [31, 1],
            "traced_tactics": [{
                "tactic": "exact add_comm",
                "annotated_tactic": ["exact <a>add_comm</a>", [{
                    "full_name": "add_comm",
                    "def_path": "Mathlib/Algebra/Basic.lean",
                    "def_pos": [10, 9],
                    "def_end_pos": [10, 17],
                }]],
                "state_before": f"goal state number {i}",
                "state_after": "no goals",
            }],
        })
    return out


def test_cap_downsamples_frequent_positive(tmp_path):
    corpus = _corpus()
    path = _write_train(tmp_path, _repeated_positive_theorems(5))

    uncapped = list(build_triplets(corpus, path, negatives="random", n_neg=1, cap=None, seed=5))
    assert len(uncapped) == 5                               # all five kept without a cap

    capped = list(build_triplets(corpus, path, negatives="random", n_neg=1, cap=2, seed=5))
    assert len(capped) == 2                                 # keep exactly the first `cap` = 2
    # capping never zeroes a premise, and is deterministic
    assert capped == list(build_triplets(corpus, path, negatives="random", n_neg=1, cap=2, seed=5))


# -- pure band-selection logic (the BM25 path, no bm25s needed) -----------------------------------

def test_select_negatives_band_skip_and_filter():
    ranked = ["a", "b", "c", "d", "e", "f", "g"]
    allacc = set(ranked)

    # exclude 'b'; skip top-2 of the remaining -> band = [d, e, f, g]
    got = select_negatives_from_ranking(
        ranked, lambda u: u in allacc, {"b"}, n_neg=2, rng=random.Random(0), skip=2, window=50
    )
    assert len(got) == 2 and set(got) <= {"d", "e", "f", "g"}

    # inaccessible candidates are filtered out; too few -> returns all accessible (order-preserving)
    got2 = select_negatives_from_ranking(
        ranked, lambda u: u in {"a", "c", "d"}, set(), n_neg=5, rng=random.Random(0),
        skip=0, window=50,
    )
    assert got2 == ["a", "c", "d"]


# -- RandomMiner ----------------------------------------------------------------------------------

def test_random_miner_respects_accessible_and_exclude():
    corpus = _corpus()
    miner = RandomMiner(corpus)
    uids = [p.uid for p in corpus.all_premises]

    picks = miner.mine("q", lambda u: True, {uids[0]}, 3, random.Random(1))
    assert len(picks) == 3 and len(set(picks)) == 3 and uids[0] not in picks

    only = {uids[1], uids[2]}
    picks2 = miner.mine("q", lambda u: u in only, set(), 5, random.Random(1))
    assert set(picks2) <= only                              # never samples the inaccessible


# -- BM25Miner smoke (needs bm25s; skipped if absent) ---------------------------------------------

def test_bm25_miner_smoke():
    pytest.importorskip("bm25s")
    corpus = _corpus()
    uid_to_premise = {p.uid: p for p in corpus.all_premises}
    miner = BM25Miner(corpus, top_n=10)
    is_acc = make_accessible_check(corpus, "Mathlib/Order/Basic.lean", (30, 1), uid_to_premise)
    exclude = {LETRANS, ADD}
    picks = miner.mine("a b : α ⊢ le_trans of chain", is_acc, exclude, 2, random.Random(0))
    assert all(u not in exclude for u in picks)             # de-noising respected
    assert all(is_acc(u) for u in picks)                    # accessible only
