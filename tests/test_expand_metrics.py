"""Hermetic tests for the offline metric-expansion (scripts/expand_metrics.py).

No JSON files, no cluster: hand-built per-example records. The core guarantee under test is that
`recompute` reproduces the SAME pure metric functions the harness used, so an extended table can be
derived from saved records without re-running retrieval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from prooflens.eval.metrics import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from expand_metrics import (  # noqa: E402
    _first_hit_rank,
    check_against_stored,
    load_eval_json,
    main,
    recompute,
)

# Three examples: a mid-rank single gold, a multi-gold hit, and a total miss.
RECORDS = [
    {"eid": 1, "gold": ["b"], "retrieved": ["a", "b", "c", "d"]},
    {"eid": 2, "gold": ["x", "z"], "retrieved": ["x", "y", "z"]},
    {"eid": 3, "gold": ["q"], "retrieved": ["m", "n"]},
]


def _mean(fn) -> float:
    return sum(fn(r["retrieved"], set(r["gold"])) for r in RECORDS) / len(RECORDS)


# -- recompute matches the harness's own pure functions (the load-bearing guarantee) --------------

def test_recompute_matches_metrics_module():
    got = recompute(RECORDS, [1, 5, 10])
    assert got["R@5"] == pytest.approx(_mean(lambda r, g: recall_at_k(r, g, 5)))
    assert got["R@10"] == pytest.approx(_mean(lambda r, g: recall_at_k(r, g, 10)))
    assert got["MRR"] == pytest.approx(_mean(reciprocal_rank))
    assert got["MAP"] == pytest.approx(_mean(average_precision))
    assert got["nDCG@10"] == pytest.approx(_mean(lambda r, g: ndcg_at_k(r, g, 10)))


def test_recompute_exact_simple_values():
    got = recompute(RECORDS, [1, 5])
    # R@1: ex1 miss(0), ex2 {x}/{x,z}=0.5, ex3 miss(0)
    assert got["R@1"] == pytest.approx((0 + 0.5 + 0) / 3)
    # first-hit ranks are 2 (ex1) and 1 (ex2); ex3 has none
    assert got["MeanRank"] == pytest.approx(1.5)
    assert got["MedianRank"] == pytest.approx(1.5)
    assert got["AnyHit"] == pytest.approx(2 / 3)


def test_new_k_values_are_available_from_records():
    # R@100 is not in the locked reporting set but recomputes fine from the top-k list
    got = recompute(RECORDS, [100])
    assert "R@100" in got and got["R@100"] is not None


# -- first-hit rank helper ------------------------------------------------------------------------

def test_first_hit_rank_dedupes_and_returns_none_on_miss():
    assert _first_hit_rank(["a", "b"], {"b"}) == 2
    assert _first_hit_rank(["a", "a", "b"], {"b"}) == 2   # dupes collapse -> b still at rank 2
    assert _first_hit_rank(["a", "z"], {"q"}) is None


# -- the correctness gate (recompute == stored aggregate) -----------------------------------------

def test_check_passes_when_stored_matches_recompute():
    got = recompute(RECORDS, [1, 10])
    stored = {"R@1": got["R@1"], "R@10": got["R@10"], "MRR": got["MRR"]}
    assert check_against_stored(stored, got) == []


def test_check_flags_a_disagreement():
    got = recompute(RECORDS, [1, 10])
    problems = check_against_stored({"R@1": got["R@1"] + 0.5}, got)
    assert problems and "R@1" in problems[0]


# -- file discovery -------------------------------------------------------------------------------

def test_load_eval_json_rejects_non_eval_files(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"provenance": {"config_name": "x"}, "examples": []}), "utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "an eval file"}), "utf-8")
    assert load_eval_json(good) is not None
    assert load_eval_json(bad) is None


def test_main_skips_non_retrieval_records_without_crashing(tmp_path, monkeypatch, capsys):
    # A retrieval eval (recompute must match its stored R@1) and a generation-style file whose
    # records lack retrieved/gold — the latter must be skipped, not crash (regression: KeyError).
    ret = tmp_path / "ret.json"
    ret.write_text(json.dumps({
        "provenance": {"config_name": "ret", "split": "random"},
        "metrics": {"R@1": 1.0},                       # a at rank 1 -> R@1 = 1.0
        "examples": [{"eid": 1, "gold": ["a"], "retrieved": ["a", "b"]}],
    }), "utf-8")
    gen = tmp_path / "gen.json"
    gen.write_text(json.dumps({
        "provenance": {"config_name": "gen", "split": "random"},
        "examples": [{"eid": 1, "match@1": 0.0}],       # no retrieved/gold
    }), "utf-8")
    monkeypatch.setattr(sys, "argv",
                        ["expand_metrics.py", "--check", "--files", str(ret), str(gen)])
    main()  # must not raise
    out = capsys.readouterr().out
    assert "not a retrieval eval" in out
    assert "PASSED" in out
