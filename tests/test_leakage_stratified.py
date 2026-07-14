"""Unit tests for the stratified leakage test (scripts/leakage_stratified.py).

The point of the script is to be **falsifiable**: it must be able to say "leakage NOT supported"
when the data says so. So the tests cover both directions — a synthetic world where leakage is real
(clean holdout scores far worse) and one where it is not (clean scores the same). If the script only
ever confirmed the hypothesis it would be worthless as evidence.

Hermetic: builds tiny split files + a metrics JSON in the shapes the real ones use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from leakage_stratified import full_name_to_paths, stratify, theorem_ids  # noqa: E402


def _write_split(path: Path, theorems: list[tuple[str, str]]) -> None:
    """theorems = [(file_path, full_name), ...] -> a split JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{"file_path": fp, "full_name": fn} for fp, fn in theorems]), encoding="utf-8"
    )


def _write_metrics(path: Path, per_example: list[tuple[str, float]]) -> None:
    """per_example = [(eid, r10), ...] -> a metrics JSON in evaluate.py's shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "provenance": {"split": "novel_premises", "model_id": "test-model"},
        "metrics": {},
        "examples": [
            {"eid": eid, "metrics": {"R@1": r10 / 2, "R@10": r10, "MRR": r10, "nDCG@10": r10}}
            for eid, r10 in per_example
        ],
    }), encoding="utf-8")


def _build(tmp_path: Path, *, clean_r10: float, leaked_r10: float) -> dict:
    """A world with 3 leaked theorems (in random/train) and 2 clean ones (not)."""
    root = tmp_path / "bench"
    _write_split(root / "random" / "train.json",
                 [("A.lean", "thm_a"), ("B.lean", "thm_b"), ("C.lean", "thm_c")])
    _write_split(root / "novel_premises" / "test.json",
                 [("A.lean", "thm_a"), ("B.lean", "thm_b"), ("C.lean", "thm_c"),
                  ("D.lean", "thm_d"), ("E.lean", "thm_e")])          # d, e are CLEAN
    _write_metrics(tmp_path / "m.json", [
        ("thm_a#0", leaked_r10), ("thm_b#0", leaked_r10), ("thm_c#0", leaked_r10),
        ("thm_d#0", clean_r10), ("thm_e#0", clean_r10),
    ])
    return stratify(str(tmp_path / "m.json"), str(root), n_boot=200)


def test_partitions_by_theorem_membership(tmp_path):
    res = _build(tmp_path, clean_r10=0.3, leaked_r10=0.9)
    assert res["n_leaked"] == 3
    assert res["n_clean"] == 2
    assert res["n_unmapped_dropped"] == 0


def test_leakage_world_shows_large_gap(tmp_path):
    """Leakage real: the model aces theorems it trained on, flops on the clean holdout."""
    res = _build(tmp_path, clean_r10=0.28, leaked_r10=0.95)
    leaked = res["groups"]["LEAKED (theorem in random/train)"]
    clean = res["groups"]["CLEAN (never trained on)"]
    assert leaked["R@10"] == pytest.approx(0.95)
    assert clean["R@10"] == pytest.approx(0.28)
    assert leaked["R@10"] - clean["R@10"] > 0.5          # a big, unambiguous gap


def test_no_leakage_world_shows_no_gap(tmp_path):
    """The falsification case: if the clean holdout scores the same, the script must NOT
    manufacture a leakage signal."""
    res = _build(tmp_path, clean_r10=0.9, leaked_r10=0.9)
    leaked = res["groups"]["LEAKED (theorem in random/train)"]
    clean = res["groups"]["CLEAN (never trained on)"]
    assert leaked["R@10"] == pytest.approx(clean["R@10"])
    assert abs(leaked["R@10"] - clean["R@10"]) < 1e-9    # no gap => hypothesis unsupported


def test_multi_tactic_eids_map_to_their_theorem(tmp_path):
    """eid is f'{full_name}#{i}' — every tactic of a theorem must land in the same group."""
    root = tmp_path / "bench"
    _write_split(root / "random" / "train.json", [("A.lean", "thm_a")])
    _write_split(root / "novel_premises" / "test.json",
                 [("A.lean", "thm_a"), ("D.lean", "thm_d")])
    _write_metrics(tmp_path / "m.json", [
        ("thm_a#0", 1.0), ("thm_a#1", 1.0), ("thm_a#2", 1.0),      # 3 tactics, all leaked
        ("thm_d#0", 0.0), ("thm_d#1", 0.0),                        # 2 tactics, both clean
    ])
    res = stratify(str(tmp_path / "m.json"), str(root), n_boot=100)
    assert res["n_leaked"] == 3
    assert res["n_clean"] == 2


def test_theorem_name_reused_across_files_is_dropped_when_ambiguous(tmp_path):
    """A full_name in two files, one in random/train and one not: identity is unrecoverable from
    the eid, so the example must be DROPPED rather than guessed — no silent miscounting."""
    root = tmp_path / "bench"
    _write_split(root / "random" / "train.json", [("A.lean", "dup")])
    _write_split(root / "novel_premises" / "test.json",
                 [("A.lean", "dup"), ("Z.lean", "dup"), ("D.lean", "thm_d")])
    _write_metrics(tmp_path / "m.json", [("dup#0", 1.0), ("thm_d#0", 0.0)])
    res = stratify(str(tmp_path / "m.json"), str(root), n_boot=100)
    assert res["n_ambiguous_dropped"] == 1
    assert res["n_leaked"] == 0
    assert res["n_clean"] == 1                                     # only thm_d survives


def test_wrong_split_is_refused(tmp_path):
    root = tmp_path / "bench"
    _write_split(root / "random" / "train.json", [("A.lean", "thm_a")])
    _write_split(root / "novel_premises" / "test.json", [("A.lean", "thm_a")])
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "provenance": {"split": "random"}, "metrics": {},
        "examples": [{"eid": "thm_a#0", "metrics": {"R@1": 1, "R@10": 1, "MRR": 1, "nDCG@10": 1}}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="novel_premises"):
        stratify(str(p), str(root))


def test_helpers_read_split_files(tmp_path):
    root = tmp_path / "bench"
    _write_split(root / "random" / "train.json", [("A.lean", "thm_a"), ("B.lean", "thm_b")])
    assert theorem_ids(str(root / "random" / "train.json")) == {
        ("A.lean", "thm_a"), ("B.lean", "thm_b")
    }
    assert full_name_to_paths(str(root / "random" / "train.json"))["thm_a"] == {"A.lean"}
