"""Tests for scripts/train_li.py (Phase 13 late-interaction fine-tuning).

The pure triplet-exploder (`load_triplet_records`) is unit-tested hermetically. The real training
step needs pylate + torch + the staged base model, so it is `skipif`-gated (runs on the cluster).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_li  # noqa: E402


def _write_jsonl(tmp_path, rows) -> str:
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def test_load_triplet_records_explodes_negatives(tmp_path):
    rows = [
        {"query": "q1", "positive": "p1", "negatives": ["n1a", "n1b"]},
        {"query": "q2", "positive": "p2", "negatives": ["n2a"]},
    ]
    out = list(train_li.load_triplet_records(_write_jsonl(tmp_path, rows)))
    assert out == [
        {"query": "q1", "positive": "p1", "negative": "n1a"},
        {"query": "q1", "positive": "p1", "negative": "n1b"},
        {"query": "q2", "positive": "p2", "negative": "n2a"},
    ]


def test_load_triplet_records_skips_empty_negatives(tmp_path):
    rows = [
        {"query": "q1", "positive": "p1", "negatives": []},          # no negative -> skipped
        {"query": "q2", "positive": "p2", "negatives": ["n2"]},
    ]
    out = list(train_li.load_triplet_records(_write_jsonl(tmp_path, rows)))
    assert out == [{"query": "q2", "positive": "p2", "negative": "n2"}]


def test_load_triplet_records_max_samples(tmp_path):
    rows = [{"query": "q", "positive": "p", "negatives": ["a", "b", "c", "d"]}]
    out = list(train_li.load_triplet_records(_write_jsonl(tmp_path, rows), max_samples=2))
    assert out == [
        {"query": "q", "positive": "p", "negative": "a"},
        {"query": "q", "positive": "p", "negative": "b"},
    ]


def test_load_triplet_records_ignores_blank_lines(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text('\n{"query":"q","positive":"p","negatives":["n"]}\n\n', encoding="utf-8")
    out = list(train_li.load_triplet_records(str(p)))
    assert out == [{"query": "q", "positive": "p", "negative": "n"}]


# -- real training smoke (cluster only) -----------------------------------------------------------

_MODELS_DIR = os.environ.get("MODELS_DIR", "")
_HAVE_PYLATE = importlib.util.find_spec("pylate") is not None


@pytest.mark.skipif(
    not (os.environ.get("PROOFLENS_TRAIN_SMOKE") == "1" and _HAVE_PYLATE and _MODELS_DIR),
    reason="real training smoke: set PROOFLENS_TRAIN_SMOKE=1 with pylate + MODELS_DIR staged",
)
def test_train_one_step_and_reload(tmp_path):
    # a handful of triplets -> 1 short training run -> checkpoint reloads into the LI retriever
    from prooflens.retrievers.late_interaction import LateInteractionRetriever

    rows = [
        {"query": f"⊢ goal {i}", "positive": f"lemma_{i} : statement {i}",
         "negatives": [f"other_{i} : different {i}"]}
        for i in range(8)
    ]
    pairs = _write_jsonl(tmp_path, rows)
    out_dir = str(tmp_path / "ckpt")
    config = {
        "name": "smoke",
        "base_model": {
            "path": str(Path(_MODELS_DIR) / "lightonai__GTE-ModernColBERT-v1"),
            "query_length": 64,
            "document_length": 64,
        },
        "train": {
            "pairs": pairs, "epochs": 1, "batch_size": 4, "lr": 1.0e-6,
            "warmup_ratio": 0.0, "bf16": False, "use_cpu": True, "seed": 42,
            "logging_steps": 1, "eval_steps": 1000, "save_steps": 1000,
        },
        "output_dir": out_dir,
    }
    train_li.train(config, limit=8)
    assert (Path(out_dir) / "training_meta.json").exists()

    retr = LateInteractionRetriever(model_path=out_dir, query_length=64, document_length=64,
                                    device="cpu")
    # a tiny corpus stand-in via the encoder path: encode a query + a doc, MaxSim is finite
    enc = retr._get_encoder()
    q = enc.encode_queries(["⊢ goal 0"])[0]
    d = enc.encode_documents(["lemma_0 : statement 0"])[0]
    assert q.shape[1] == d.shape[1] and q.shape[0] > 0 and d.shape[0] > 0
