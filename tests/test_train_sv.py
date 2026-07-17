"""Tests for scripts/train_sv.py — the MATCHED single-vector control (Phase 15).

The control's entire scientific value is the sentence "everything is identical to the LI run except
the matching mechanism". That sentence is enforced here as executable invariants over the config
files, so a silent edit to either side breaks the suite instead of quietly un-matching the
experiment:

- **Matched-control parity:** per split, the SV train config uses the SAME pairs/val files, epochs,
  batch size and seed as the LI train config. The LRs deliberately differ (each architecture's
  best practice — documented in integrity_notes.md); their exact values are pinned so a change is
  a conscious act that updates this test.
- **Split-matching:** each eval config pins exactly its own split; each train config consumes its
  own split's pairs.
- **Frozen-harness wiring:** the eval configs route through `retriever: dense` with the
  `sentence_transformer` encoder and `full_name_code` premise text, and score the checkpoint the
  train config produces (paths agree).

The real 1-step training smoke is `skipif`-gated (cluster: sentence-transformers + staged base
model). Run it INSIDE an allocation, not on the login node (integrity_notes.md).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_sv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    with open(ROOT / rel, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# -- matched-control parity (the invariant the whole comparison rests on) -------------------------

# The SV control's canonical configs are the sweep-chosen lr=3e-6 files (the initial 2e-5 run
# catastrophically damaged the model — see results/phase_logs/phase15.md; the 2e-5 configs were
# removed). The winning lr turned out to EQUAL LI's (3e-6), so the control is now matched on lr too.
@pytest.mark.parametrize("li_cfg, sv_cfg", [
    ("configs/train/li_ft_random.yaml", "configs/train/sv_ft_random_lr3e6.yaml"),
    ("configs/train/li_ft_novel.yaml", "configs/train/sv_ft_novel_lr3e6.yaml"),
])
def test_sv_and_li_train_on_identical_data_and_budget(li_cfg, sv_cfg):
    li, sv = _load(li_cfg)["train"], _load(sv_cfg)["train"]
    # SAME data: triplets and val file are literally the same paths
    assert sv["pairs"] == li["pairs"]
    assert sv["val_pairs"] == li["val_pairs"]
    # SAME budget: epochs, batch (=> same in-batch-negative count), seed
    assert sv["epochs"] == li["epochs"]
    assert sv["batch_size"] == li["batch_size"]
    assert sv["seed"] == li["seed"]
    assert sv["warmup_ratio"] == li["warmup_ratio"]
    # SAME lr now too: the sweep selected 3e-6, equal to LI's -> an even tighter matched control.
    # Pinned so any re-tune is a conscious, test-updating act. (2e-5 damaged the model; 1e-5 barely
    # beat the untrained base; 3e-6 won at 32.0 R@10 vs base 11.4 — phase15.md.)
    assert float(sv["lr"]) == pytest.approx(3.0e-6)
    assert float(li["lr"]) == pytest.approx(3.0e-6)
    assert float(sv["lr"]) == float(li["lr"])


def test_sv_base_is_the_li_models_lineage():
    # gte-modernbert-base is the ModernBERT that GTE-ModernColBERT is built on -> the comparison
    # holds the base lineage fixed and varies only the matching head.
    for cfg in ("configs/train/sv_ft_random_lr3e6.yaml", "configs/train/sv_ft_novel_lr3e6.yaml"):
        base = _load(cfg)["base_model"]
        assert base["hf_id"] == "Alibaba-NLP/gte-modernbert-base"
        assert "gte-modernbert-base" in base["path"]


# -- split-matching + frozen-harness wiring of the eval configs -----------------------------------

@pytest.mark.parametrize("train_cfg, eval_cfg, split, pairs_token", [
    ("configs/train/sv_ft_random_lr3e6.yaml", "configs/dense_sv_ft_random_lr3e6.yaml",
     "random", "random_bm25_train"),
    ("configs/train/sv_ft_novel_lr3e6.yaml", "configs/dense_sv_ft_novel_lr3e6.yaml",
     "novel_premises", "novel_premises_bm25_train"),
])
def test_sv_eval_config_scores_the_trained_checkpoint_split_matched(
        train_cfg, eval_cfg, split, pairs_token):
    tr, ev = _load(train_cfg), _load(eval_cfg)
    # split-matched: trained on its own split's pairs, evaluated ONLY on that split's test set
    assert pairs_token in tr["train"]["pairs"]
    assert ev["eval"]["splits"] == [split]
    # the eval scores exactly the checkpoint the training writes
    assert ev["model"]["path"] == tr["output_dir"]
    # through the frozen dense harness, with the control's encoder + LI-matched premise text
    assert ev["retriever"] == "dense"
    assert ev["model"]["encoder"] == "sentence_transformer"
    assert ev["model"]["premise_text"] == "full_name_code"
    # train and eval agree on sequence length (train/eval consistency within the architecture)
    assert ev["model"]["max_length"] == tr["base_model"]["max_length"]
    # a FRESH index dir (the SV embeddings are new; reusing another index would be silently wrong)
    assert ev["index"]["dir"].rstrip("/").endswith(Path(tr["output_dir"]).name)


def test_the_two_sv_runs_do_not_share_checkpoints_or_indices():
    r = _load("configs/dense_sv_ft_random_lr3e6.yaml")
    n = _load("configs/dense_sv_ft_novel_lr3e6.yaml")
    assert r["model"]["path"] != n["model"]["path"]
    assert r["index"]["dir"] != n["index"]["dir"]


# -- small pure helpers ----------------------------------------------------------------------------

def test_read_meta_returns_sidecar(tmp_path):
    pairs = tmp_path / "x_train.jsonl"
    pairs.write_text("{}", encoding="utf-8")
    (tmp_path / "x_train.meta.json").write_text('{"seed": 42}', encoding="utf-8")
    assert train_sv._read_meta(str(pairs)) == {"seed": 42}


def test_read_meta_missing_or_corrupt_is_none(tmp_path):
    pairs = tmp_path / "y_train.jsonl"
    pairs.write_text("{}", encoding="utf-8")
    assert train_sv._read_meta(str(pairs)) is None            # no sidecar
    (tmp_path / "y_train.meta.json").write_text("{not json", encoding="utf-8")
    assert train_sv._read_meta(str(pairs)) is None            # corrupt sidecar


def test_expand_expands_env_vars(monkeypatch):
    monkeypatch.setenv("PL_TEST_SCRATCH", "/tmp/scr")
    assert train_sv._expand("${PL_TEST_SCRATCH}/x") == "/tmp/scr/x"
    assert train_sv._expand(42) == 42                          # non-str passes through


# -- real training smoke (cluster only; run INSIDE an allocation, not the login node) -------------

_MODELS_DIR = os.environ.get("MODELS_DIR", "")
_HAVE_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not (os.environ.get("PROOFLENS_SV_TRAIN_SMOKE") == "1" and _HAVE_ST and _MODELS_DIR),
    reason="real SV training smoke: set PROOFLENS_SV_TRAIN_SMOKE=1 with sentence-transformers "
           "+ MODELS_DIR staged (run inside an allocation)",
)
def test_sv_train_one_step_and_reload_through_eval_path(tmp_path):
    """8 triplets -> a short real training run -> the checkpoint must reload through the EXACT
    loader the frozen harness uses (`dense._STEncoder`) and produce unit-norm vectors. This is the
    train->eval interface check — the thing that can silently break."""
    from prooflens.retrievers.dense import _STEncoder

    rows = [
        {"query": f"⊢ goal {i}", "positive": f"lemma_{i} : statement {i}",
         "negatives": [f"other_{i} : different {i}"]}
        for i in range(8)
    ]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_dir = str(tmp_path / "ckpt")
    config = {
        "name": "sv_smoke",
        "base_model": {
            "path": str(Path(_MODELS_DIR) / "Alibaba-NLP__gte-modernbert-base"),
            "max_length": 64,
        },
        "train": {
            "pairs": str(pairs), "epochs": 1, "batch_size": 4, "lr": 1.0e-6,
            "warmup_ratio": 0.0, "bf16": False, "use_cpu": True, "seed": 42,
            "logging_steps": 1, "eval_steps": 1000, "save_steps": 1000,
        },
        "output_dir": out_dir,
    }
    train_sv.train(config, limit=8)

    meta = json.loads((Path(out_dir) / "training_meta.json").read_text(encoding="utf-8"))
    assert meta["n_train_triplets"] == 8
    assert meta["architecture"].startswith("single-vector")

    import numpy as np
    enc = _STEncoder(out_dir, max_length=64, batch_size=4, device="cpu")
    embs = enc.encode(["⊢ goal 0", "lemma_0 : statement 0"])
    assert embs.shape == (2, enc.dim)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)   # unit-norm (eval contract)
    two = enc.encode(["same text", "same text"])
    assert float(two[0] @ two[1]) == pytest.approx(1.0, abs=1e-4)      # identical text -> cosine 1
