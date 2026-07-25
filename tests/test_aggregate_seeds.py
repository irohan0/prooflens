"""Hermetic tests for the per-seed aggregator (scripts/aggregate_seeds.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from aggregate_seeds import aggregate  # noqa: E402


def _write(tmp_path: Path, name: str, metrics: dict) -> str:
    p = tmp_path / name
    p.write_text(json.dumps({"provenance": {}, "metrics": metrics}), encoding="utf-8")
    return str(p)


def test_mean_and_sample_std_across_seeds(tmp_path):
    files = [
        _write(tmp_path, "s0.json", {"R@10": 0.20, "MRR": 0.10}),
        _write(tmp_path, "s1.json", {"R@10": 0.30, "MRR": 0.20}),
        _write(tmp_path, "s2.json", {"R@10": 0.40, "MRR": 0.30}),
    ]
    agg = aggregate(files, ["R@10", "MRR"])
    mean, std, n = agg["R@10"]
    assert mean == pytest.approx(0.30)
    assert std == pytest.approx(0.1)           # sample std of [.2,.3,.4] = 0.1
    assert n == 3


def test_missing_metric_in_one_file_is_skipped_not_crashed(tmp_path):
    files = [
        _write(tmp_path, "a.json", {"R@10": 0.20, "MAP": 0.15}),
        _write(tmp_path, "b.json", {"R@10": 0.40}),          # no MAP here
    ]
    agg = aggregate(files, ["R@10", "MAP"])
    assert agg["R@10"][2] == 2                  # both files contribute R@10
    assert agg["MAP"][2] == 1                   # only one contributes MAP
    assert agg["MAP"][1] == 0.0                 # std undefined for n=1 -> 0.0


def test_single_file_has_zero_std(tmp_path):
    files = [_write(tmp_path, "only.json", {"R@1": 0.085})]
    agg = aggregate(files, ["R@1"])
    assert agg["R@1"] == pytest.approx((0.085, 0.0, 1))
