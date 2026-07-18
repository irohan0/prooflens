"""Unit tests for the structural-vs-lexical stratification (scripts/lexical_stratified.py).

The script must (a) bucket novel examples by the audit's exact gold-name-in-state rule, (b) compare
LI vs SV example-paired within each bucket, and (c) be able to report EITHER "structural advantage
survives" OR "advantage is largely lexical" — it must not be rigged to only confirm the thesis. Both
directions are covered, plus a fixture-backed check that the context builder loads real data and
computes the lexical flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import lexical_stratified as ls  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


def _write_records(path: Path, split: str, per_example: dict[str, dict]) -> None:
    """per_example: {eid: {"R@1":.., "R@10":.., "hit_ranks":{g:rank}}} -> a metrics JSON."""
    examples = []
    for eid, d in per_example.items():
        examples.append({
            "eid": eid,
            "hit_ranks": d.get("hit_ranks", {}),
            "metrics": {"R@1": d["R@1"], "R@10": d["R@10"], "MRR": d.get("MRR", d["R@1"]),
                        "nDCG@10": d.get("R@10", 0.0)},
        })
    path.write_text(json.dumps({"provenance": {"split": split}, "examples": examples}),
                    encoding="utf-8")


# -- the audit's gold-name-in-state rule ----------------------------------------------------------

def test_gold_name_in_state_true_and_false():
    # a uid is "{path}::{full_name}@line,col"; short_name is the last dotted component
    gold = ["Mathlib/X.lean::Nat.add_comm@10,1"]
    assert ls.gold_name_in_state("a b : Nat ⊢ add_comm a b", gold) is True
    assert ls.gold_name_in_state("a b : Nat ⊢ a + b = b + a", gold) is False


# -- paired delta + bootstrap CI ------------------------------------------------------------------

def test_paired_delta_positive_when_li_beats_sv():
    li = [{"metrics": {"R@10": 1.0}} for _ in range(80)]
    sv = [{"metrics": {"R@10": 0.0}} for _ in range(80)]
    delta, lo, hi = ls._paired_delta_ci(li, sv, "R@10", 2000, np.random.default_rng(0))
    assert delta == pytest.approx(1.0)
    assert lo > 0                                  # CI clearly excludes zero

def test_paired_delta_zero_when_tied():
    same = [{"metrics": {"R@10": v}} for v in (0.0, 1.0, 0.0, 1.0)]
    delta, lo, hi = ls._paired_delta_ci(same, list(same), "R@10", 500, np.random.default_rng(0))
    assert delta == 0.0
    assert lo == 0.0 and hi == 0.0


# -- bucketing + aggregation (monkeypatch the data-loading context) -------------------------------

def _run(tmp_path, ctx, li_recs, sv_recs, monkeypatch, seed=42):
    monkeypatch.setattr(ls, "build_eid_context", lambda *a, **k: ctx)
    li_p, sv_p = tmp_path / "li.json", tmp_path / "sv.json"
    _write_records(li_p, "novel_premises", li_recs)
    _write_records(sv_p, "novel_premises", sv_recs)
    return ls.stratify(str(li_p), str(sv_p), "corpus", "splits", "novel_premises", n_boot=1000)


def test_buckets_split_by_lexical_flag(tmp_path, monkeypatch):
    ctx = {
        "t#0": {"lexical": True, "gold_names": ["add_comm"]},
        "t#1": {"lexical": True, "gold_names": ["mul_comm"]},
        "t#2": {"lexical": False, "gold_names": ["foo"]},
    }
    recs = {e: {"R@1": 0.0, "R@10": 1.0} for e in ctx}
    res = _run(tmp_path, ctx, recs, recs, monkeypatch)
    assert res["buckets"]["LEXICAL (gold name in state)"]["n"] == 2
    assert res["buckets"]["STRUCTURAL (gold name NOT in state)"]["n"] == 1


def test_structural_advantage_detected(tmp_path, monkeypatch):
    """The thesis-supporting world: in the STRUCTURAL bucket LI hits, SV misses -> positive,
    significant LI-SV; verdict says 'not a neural BM25'."""
    ctx = {f"s{i}#0": {"lexical": False, "gold_names": [f"lem{i}"]} for i in range(60)}
    ctx.update({f"l{i}#0": {"lexical": True, "gold_names": [f"kem{i}"]} for i in range(60)})
    li = {e: {"R@1": 0.0, "R@10": 1.0, "hit_ranks": {"g": 3}} for e in ctx}
    sv = {}
    for e in ctx:
        hit = 1.0 if ctx[e]["lexical"] else 0.0     # SV only hits when the name is in the state
        sv[e] = {"R@1": 0.0, "R@10": hit, "hit_ranks": {"g": 5 if hit else None}}
    res = _run(tmp_path, ctx, li, sv, monkeypatch)
    struct = res["buckets"]["STRUCTURAL (gold name NOT in state)"]["R@10"]
    assert struct["delta"] == pytest.approx(1.0)    # LI 1.0 - SV 0.0
    assert struct["significant"] is True
    assert struct["ci"][0] > 0
    # 60 structural examples where LI hit top-10 and SV missed -> all surfaced as wins
    assert res["structural_li_wins"]["n_total"] == 60


def test_advantage_is_only_lexical_is_reported_honestly(tmp_path, monkeypatch):
    """The falsification world: LI and SV are identical on structural examples (LI's edge is only in
    the lexical bucket). The structural delta must be ~0 and NOT flagged significant."""
    ctx = {f"s{i}#0": {"lexical": False, "gold_names": ["x"]} for i in range(50)}
    recs = {e: {"R@1": 0.0, "R@10": float(i % 2)} for i, e in enumerate(ctx)}
    res = _run(tmp_path, ctx, recs, dict(recs), monkeypatch)   # LI == SV on structural
    struct = res["buckets"]["STRUCTURAL (gold name NOT in state)"]["R@10"]
    assert struct["delta"] == 0.0
    assert struct["significant"] is False
    assert res["structural_li_wins"]["n_total"] == 0           # no LI-only structural wins


def test_structural_wins_ranked_by_li_rank(tmp_path, monkeypatch):
    ctx = {
        "a#0": {"lexical": False, "gold_names": ["alpha"]},
        "b#0": {"lexical": False, "gold_names": ["beta"]},
    }
    li = {
        "a#0": {"R@1": 0.0, "R@10": 1.0, "hit_ranks": {"g": 7}},
        "b#0": {"R@1": 1.0, "R@10": 1.0, "hit_ranks": {"g": 2}},
    }
    sv = {e: {"R@1": 0.0, "R@10": 0.0, "hit_ranks": {"g": None}} for e in ctx}
    res = _run(tmp_path, ctx, li, sv, monkeypatch)
    wins = res["structural_li_wins"]["examples"]
    assert [w["eid"] for w in wins] == ["b#0", "a#0"]          # sorted by LI best rank (2 before 7)
    assert wins[0]["li_best_rank"] == 2 and wins[0]["sv_best_rank"] is None


def test_split_mismatch_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "build_eid_context", lambda *a, **k: {})
    li_p, sv_p = tmp_path / "li.json", tmp_path / "sv.json"
    _write_records(li_p, "random", {"t#0": {"R@1": 0.0, "R@10": 1.0}})
    _write_records(sv_p, "novel_premises", {"t#0": {"R@1": 0.0, "R@10": 1.0}})
    with pytest.raises(ValueError, match="split"):
        ls.stratify(str(li_p), str(sv_p), "c", "s", "novel_premises")


# -- fixture-backed context builder (exercises real corpus + split loading) -----------------------

def test_build_eid_context_on_fixture():
    ctx = ls.build_eid_context(str(FIX / "corpus.jsonl"), str(FIX), "random")
    assert ctx, "fixture split produced no examples"
    for v in ctx.values():
        assert isinstance(v["lexical"], bool)
        assert isinstance(v["gold_names"], list) and v["gold_names"]
