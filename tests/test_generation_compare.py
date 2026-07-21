"""Hermetic tests for the Phase-21 condition comparison.

Builds tiny synthetic generation JSONs (the shape eval/generate_eval.py writes), so this needs no
cluster results. The properties asserted are the ones that keep the final table honest: it must
refuse to mix splits or a subset run with a full run, it must find the no-premises floor, and the
paired statistics must actually detect a real effect while not inventing one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generation_compare import (  # noqa: E402
    check_comparable,
    find_baseline,
    load_run,
    print_paired,
    verdict,
)

METRICS = ["match@1", "premise_name@1"]


def _write_run(path: Path, name: str, source: str, per_example: list[dict],
               split: str = "novel_premises", is_full: bool = True) -> str:
    blob = {
        "provenance": {
            "config_name": name,
            "split": split,
            "n_examples": len(per_example),
            "is_full_run": is_full,
            "seconds_per_example": 2.5,
            "premise_condition": {"source": source, "retriever_config_name": f"{name}_retriever"},
        },
        "metrics": {
            m: sum(r["metrics"][m] for r in per_example) / len(per_example) for m in METRICS
        },
        "examples": per_example,
    }
    path.write_text(json.dumps(blob), encoding="utf-8")
    return str(path)


def _examples(values: list[tuple[float, float]]) -> list[dict]:
    return [
        {
            "eid": f"thm#{i}",
            "n_premises_in_context": 20,
            "metrics": {"match@1": m, "premise_name@1": p},
        }
        for i, (m, p) in enumerate(values)
    ]


def test_load_run_extracts_the_summary(tmp_path):
    p = _write_run(tmp_path / "a.json", "gen_none_novel", "none", _examples([(0.0, 1.0)] * 4))
    r = load_run(p)
    assert r["name"] == "gen_none_novel"
    assert r["source"] == "none"
    assert r["n"] == 4
    assert r["mean_in_context"] == 20.0
    assert r["metrics"]["premise_name@1"] == 1.0


def test_find_baseline_picks_the_no_premises_run(tmp_path):
    a = load_run(_write_run(tmp_path / "a.json", "gen_bm25_novel", "retrieval_json",
                            _examples([(0.0, 0.0)] * 3)))
    b = load_run(_write_run(tmp_path / "b.json", "gen_none_novel", "none",
                            _examples([(0.0, 0.0)] * 3)))
    assert find_baseline([a, b])["name"] == "gen_none_novel"


def test_find_baseline_returns_none_when_absent(tmp_path):
    a = load_run(_write_run(tmp_path / "a.json", "gen_bm25_novel", "retrieval_json",
                            _examples([(0.0, 0.0)] * 3)))
    assert find_baseline([a]) is None


def test_refuses_to_mix_splits(tmp_path):
    a = load_run(_write_run(tmp_path / "a.json", "a", "none", _examples([(0.0, 0.0)] * 3),
                            split="random"))
    b = load_run(_write_run(tmp_path / "b.json", "b", "retrieval_json",
                            _examples([(0.0, 0.0)] * 3), split="novel_premises"))
    with pytest.raises(ValueError, match="multiple splits"):
        check_comparable([a, b])


def test_refuses_to_mix_subset_and_full_runs(tmp_path):
    a = load_run(_write_run(tmp_path / "a.json", "a", "none", _examples([(0.0, 0.0)] * 3)))
    b = load_run(_write_run(tmp_path / "b.json", "b", "retrieval_json",
                            _examples([(0.0, 0.0)] * 5)))
    with pytest.raises(ValueError, match="different example counts"):
        check_comparable([a, b])


def test_warns_about_subset_runs(tmp_path, capsys):
    a = load_run(_write_run(tmp_path / "a.json", "pilot", "none",
                            _examples([(0.0, 0.0)] * 3), is_full=False))
    check_comparable([a])
    assert "WARNING" in capsys.readouterr().out


def test_paired_comparison_detects_a_real_effect(tmp_path):
    # baseline misses everything; the comparison hits almost everything -> must be significant
    base = load_run(_write_run(tmp_path / "base.json", "floor", "none",
                               _examples([(0.0, 0.0)] * 40)))
    good = load_run(_write_run(tmp_path / "good.json", "ours", "retrieval_json",
                               _examples([(1.0, 1.0)] * 38 + [(0.0, 0.0)] * 2)))
    rows = print_paired(base, good, METRICS, n_boot=2000, n_perm=2000, seed=42)
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["match@1"]["delta"] > 0
    assert by_metric["match@1"]["significant"] is True
    assert by_metric["match@1"]["ci_low"] > 0


def test_paired_comparison_does_not_invent_an_effect(tmp_path):
    # identical per-example outcomes -> zero delta, never significant
    vals = _examples([(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (0.0, 0.0)] * 10)
    a = load_run(_write_run(tmp_path / "a.json", "a", "none", vals))
    b = load_run(_write_run(tmp_path / "b.json", "b", "retrieval_json", vals))
    rows = print_paired(a, b, METRICS, n_boot=2000, n_perm=2000, seed=42)
    for r in rows:
        assert r["delta"] == 0.0
        assert r["significant"] is False


def test_verdict_requires_both_ci_and_p():
    """A CI that excludes zero is NOT enough if the permutation test disagrees."""
    # the exact dry-run case: CI excludes zero, p = 0.0625 -> must not print SIGNIFICANT
    assert verdict({"significant": True, "p_value": 0.0625}) == "borderline"
    assert verdict({"significant": True, "p_value": 0.0005}) == "SIGNIFICANT"
    assert verdict({"significant": False, "p_value": 0.01}) == "borderline"
    assert verdict({"significant": False, "p_value": 0.9}) == "ns"


def test_verdict_is_attached_to_each_row(tmp_path):
    base = load_run(_write_run(tmp_path / "base.json", "floor", "none",
                               _examples([(0.0, 0.0)] * 40)))
    good = load_run(_write_run(tmp_path / "good.json", "ours", "retrieval_json",
                               _examples([(1.0, 1.0)] * 38 + [(0.0, 0.0)] * 2)))
    rows = print_paired(base, good, METRICS, n_boot=2000, n_perm=2000, seed=42)
    assert all(r["verdict"] in {"SIGNIFICANT", "borderline", "ns"} for r in rows)


def test_paired_comparison_is_reproducible(tmp_path):
    base = load_run(_write_run(tmp_path / "base.json", "floor", "none",
                               _examples([(0.0, 0.0)] * 30)))
    good = load_run(_write_run(tmp_path / "good.json", "ours", "retrieval_json",
                               _examples([(1.0, 1.0)] * 20 + [(0.0, 0.0)] * 10)))
    r1 = print_paired(base, good, METRICS, n_boot=1000, n_perm=1000, seed=7)
    r2 = print_paired(base, good, METRICS, n_boot=1000, n_perm=1000, seed=7)
    assert [r["ci_low"] for r in r1] == [r["ci_low"] for r in r2]
    assert [r["p_value"] for r in r1] == [r["p_value"] for r in r2]
