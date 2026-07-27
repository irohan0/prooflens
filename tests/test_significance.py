"""Unit tests for the paired significance test (scripts/significance.py).

Hermetic: builds tiny synthetic metrics JSONs (the same shape eval/evaluate.py writes) so the test
needs no cluster results. Covers the properties we actually rely on when reporting a p-value:
pairing by eid, a real effect being detected, a null effect NOT being called significant, the
split-mismatch guard, and reproducibility under a fixed seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from significance import (  # noqa: E402
    bootstrap_ci,
    compare,
    load_per_example,
    paired_deltas,
    permutation_p,
)

METRICS = ["R@1", "R@10"]


def _write_run(path: Path, split: str, per_example: dict[str, dict[str, float]]) -> None:
    """Write a metrics JSON in evaluate.py's shape."""
    blob = {
        "provenance": {"split": split, "config_name": path.stem},
        "metrics": {},
        "examples": [{"eid": eid, "metrics": m} for eid, m in per_example.items()],
    }
    path.write_text(json.dumps(blob), encoding="utf-8")


def test_load_and_pair_by_eid(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    # deliberately different eid ORDER in b -> pairing must be by eid, not by position
    _write_run(a, "random", {"e1": {"R@1": 0.0}, "e2": {"R@1": 1.0}})
    _write_run(b, "random", {"e2": {"R@1": 1.0}, "e1": {"R@1": 1.0}})

    _prov, pa = load_per_example(str(a))
    _prov, pb = load_per_example(str(b))
    d = paired_deltas(pa, pb, "R@1")
    assert list(d) == [1.0, 0.0]          # eid-sorted: e1 improved, e2 unchanged


def test_disjoint_examples_raises(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {"e1": {"R@1": 1.0}})
    _write_run(b, "random", {"zz": {"R@1": 1.0}})
    _prov, pa = load_per_example(str(a))
    _prov, pb = load_per_example(str(b))
    with pytest.raises(ValueError):
        paired_deltas(pa, pb, "R@1")


def test_split_mismatch_is_refused(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {"e1": {"R@1": 1.0}})
    _write_run(b, "novel_premises", {"e1": {"R@1": 1.0}})
    with pytest.raises(ValueError, match="different splits"):
        compare(str(a), str(b), ["R@1"], n_boot=10, n_perm=10)


def test_significance_requires_both_ci_and_p(monkeypatch, tmp_path):
    """A CI excluding zero is NOT sufficient: p must clear 0.05 too, else 'borderline'.

    Regression guard — the original code used `lo > 0 or hi < 0` alone, which mislabelled a
    marginal result (CI barely excluding zero, p = 0.053) as SIGNIFICANT.
    """
    import significance as sig

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {"e1": {"R@1": 0.0}, "e2": {"R@1": 0.0}})
    _write_run(b, "random", {"e1": {"R@1": 1.0}, "e2": {"R@1": 1.0}})

    # force the marginal corner: CI excludes zero, p just above the threshold
    monkeypatch.setattr(sig, "bootstrap_ci", lambda *_a, **_k: (0.01, 0.5))
    monkeypatch.setattr(sig, "permutation_p", lambda *_a, **_k: 0.0534)
    row = sig.compare(str(a), str(b), ["R@1"], n_boot=10, n_perm=10)[0]
    assert row["significant"] is False        # p fails -> not a significance claim
    assert row["borderline"] is True

    # and when both agree, it IS significant
    monkeypatch.setattr(sig, "permutation_p", lambda *_a, **_k: 0.001)
    row = sig.compare(str(a), str(b), ["R@1"], n_boot=10, n_perm=10)[0]
    assert row["significant"] is True
    assert row["borderline"] is False


def test_real_effect_is_detected(tmp_path):
    """B beats A on 15% of 2000 examples and never loses -> must be flagged significant."""
    rng = np.random.default_rng(0)
    n = 2000
    a_vals = rng.integers(0, 2, n).astype(float)          # baseline hits ~50%
    b_vals = a_vals.copy()
    flip = rng.choice(np.flatnonzero(a_vals == 0), size=150, replace=False)
    b_vals[flip] = 1.0                                     # B fixes 150 of A's misses

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(a_vals)})
    _write_run(b, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(b_vals)})

    (row,) = compare(str(a), str(b), ["R@1"], n_boot=2000, n_perm=2000)
    assert row["delta"] == pytest.approx(150 / n)
    assert row["significant"] is True
    assert row["ci_low"] > 0                # CI excludes zero
    assert row["p_value"] < 0.01
    assert (row["wins"], row["losses"]) == (150, 0)


def test_null_effect_is_not_called_significant(tmp_path):
    """Identical systems -> zero delta, p=1, and never 'significant' (guards false positives)."""
    rng = np.random.default_rng(1)
    vals = rng.integers(0, 2, 500).astype(float)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(vals)})
    _write_run(b, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(vals)})

    (row,) = compare(str(a), str(b), ["R@1"], n_boot=1000, n_perm=1000)
    assert row["delta"] == 0.0
    assert row["significant"] is False
    assert row["p_value"] == 1.0            # all ties -> no information -> p = 1


def test_symmetric_noise_is_not_significant(tmp_path):
    """B wins as often as it loses -> a nonzero but null difference must NOT be significant."""
    rng = np.random.default_rng(7)
    n = 1500
    a_vals = rng.integers(0, 2, n).astype(float)
    b_vals = a_vals.copy()
    idx = rng.permutation(n)
    b_vals[idx[:100]] = 1.0 - b_vals[idx[:100]]   # flip 100 at random, both directions

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(a_vals)})
    _write_run(b, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(b_vals)})

    (row,) = compare(str(a), str(b), ["R@1"], n_boot=2000, n_perm=2000)
    assert row["significant"] is False
    assert row["p_value"] > 0.05


def test_seeded_and_reproducible(tmp_path):
    rng = np.random.default_rng(3)
    a_vals = rng.integers(0, 2, 300).astype(float)
    b_vals = np.clip(a_vals + rng.integers(0, 2, 300) * 0.5, 0, 1)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_run(a, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(a_vals)})
    _write_run(b, "random", {f"e{i}": {"R@1": v} for i, v in enumerate(b_vals)})

    r1 = compare(str(a), str(b), ["R@1"], n_boot=500, n_perm=500, seed=42)
    r2 = compare(str(a), str(b), ["R@1"], n_boot=500, n_perm=500, seed=42)
    assert r1 == r2                          # identical seed -> identical CI and p


def test_permutation_p_never_zero():
    """The +1/(n+1) correction: even an overwhelming effect must not report p == 0."""
    d = np.ones(200)                         # B wins every single example
    p = permutation_p(d, n_perm=100, rng=np.random.default_rng(0))
    assert 0 < p <= 1 / 101 + 1e-9


def test_bootstrap_ci_brackets_the_mean():
    d = np.concatenate([np.ones(60), np.zeros(40)])       # mean 0.6
    lo, hi = bootstrap_ci(d, n_boot=2000, rng=np.random.default_rng(0))
    assert lo < d.mean() < hi
    assert lo > 0                                          # clearly positive effect
