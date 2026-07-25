"""Hermetic tests for the seed-config generator (scripts/make_seed_configs.py).

The generator must change ONLY the seed and the per-seed output paths, so the multi-seed sweep stays
a matched control. A wrong path here would silently point an eval at the wrong checkpoint/index, so
the field rules are pinned explicitly, plus a smoke test over the real base configs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from make_seed_configs import BASE_EVAL, BASE_TRAIN, make_seed_config  # noqa: E402

TRAIN = {
    "name": "li_ft_novel",
    "base_model": {"hf_id": "x", "query_length": 384},
    "train": {"seed": 42, "epochs": 1, "pairs": "p.jsonl"},
    "output_dir": "${SCRATCH}/prooflens/checkpoints/li_ft_novel_bm25",
}
EVAL = {
    "name": "late_interaction_ft_novel_idf",
    "retriever": "late_interaction",
    "model": {"path": "${SCRATCH}/prooflens/checkpoints/li_ft_novel_bm25", "query_length": 384},
    "index": {"dir": "${SCRATCH}/prooflens/indices/li_ft_novel", "backend": "exact-maxsim"},
    "symbol_weighting": {"enabled": True, "mode": "idf",
                         "idf_path": "${SCRATCH}/prooflens/indices/token_idf.json"},
    "eval": {"splits": ["novel_premises"], "retrieve_k": 100},
    "seed": 42,
}


def test_train_config_sets_seed_and_suffixes_output_dir():
    out = make_seed_config(TRAIN, 3)
    assert out["name"] == "li_ft_novel_s3"
    assert out["train"]["seed"] == 3
    assert out["output_dir"].endswith("/li_ft_novel_bm25_s3")
    assert TRAIN["train"]["seed"] == 42                 # base not mutated (deep copy)


def test_eval_config_suffixes_model_and_index_only():
    out = make_seed_config(EVAL, 2)
    assert out["name"] == "late_interaction_ft_novel_idf_s2"
    assert out["seed"] == 2
    assert out["model"]["path"].endswith("/li_ft_novel_bm25_s2")
    assert out["index"]["dir"].endswith("/li_ft_novel_s2")
    # shared, split-agnostic table must NOT be per-seed, and eval settings are untouched
    assert out["symbol_weighting"]["idf_path"].endswith("/token_idf.json")
    assert out["eval"]["splits"] == ["novel_premises"]


def test_eval_points_at_the_checkpoint_its_train_produces():
    # base train output_dir and base eval model.path share a stem -> same suffix keeps them aligned
    assert make_seed_config(TRAIN, 5)["output_dir"] == make_seed_config(EVAL, 5)["model"]["path"]


def test_rejects_config_that_is_neither_train_nor_eval():
    with pytest.raises(ValueError, match="neither a train nor an eval"):
        make_seed_config({"name": "weird"}, 1)


def test_real_base_configs_classify_and_stamp():
    root = Path(__file__).parent.parent
    for rel in BASE_TRAIN + BASE_EVAL:
        base = yaml.safe_load((root / rel).read_text(encoding="utf-8"))
        out = make_seed_config(base, 7)
        assert out["name"].endswith("_s7")
        if "output_dir" in base:
            assert out["train"]["seed"] == 7
            assert out["output_dir"].endswith("_s7")
        else:
            assert out["seed"] == 7
            assert out["model"]["path"].endswith("_s7")
            assert out["index"]["dir"].endswith("_s7")
