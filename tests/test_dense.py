"""Unit tests for the dense ReProver retriever.

Two layers:
- **Hermetic** (numpy only, no torch): a deterministic fake encoder is injected so the whole
  retrieval path — ReProver premise serialization, accessible masking, exact-cosine top-k,
  persistence roundtrip — is exercised without downloading the ByT5 checkpoint.
- **Real-checkpoint smoke** (opt-in): guarded by `PROOFLENS_DENSE_SMOKE=1` **and** importable
  torch/transformers, so it runs on the cluster (where the checkpoint is staged offline) and is
  skipped locally/CI. It verifies the real encoder loads, pools, normalizes, and that identical
  text gives cosine 1 — the encode path itself, before the full GPU calibration run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from prooflens.data.corpus import load_corpus
from prooflens.retrievers.dense import DenseRetriever, serialize_premise

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


# -- deterministic fake encoder (stand-in for the ByT5 model) ---------------------------------

class FakeEncoder:
    """Maps each distinct string to a fixed pseudo-random L2-normalized vector. Deterministic
    across processes (sha1-seeded, not Python's salted hash), so identical text -> identical
    vector -> cosine 1, and the persistence roundtrip is reproducible."""

    def __init__(self, dim: int = 48) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha1(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            n = float(np.linalg.norm(v))
            vecs[i] = v / (n if n > 0 else 1.0)
        return vecs


class CountingEncoder(FakeEncoder):
    """FakeEncoder that records how many times encode() was called — lets a test assert that a
    reloaded index does NOT re-encode the premises."""

    def __init__(self, dim: int = 48) -> None:
        super().__init__(dim)
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return super().encode(texts)


# -- ReProver premise serialization (must match common.py verbatim) ---------------------------

def test_serialize_marks_root_reference():
    # explicit `_root_.<full_name>` form is replaced by <a>...</a>
    out = serialize_premise("Nat.add", "lemma foo : _root_.Nat.add x y = x := by rfl")
    assert out == "lemma foo : <a>Nat.add</a> x y = x := by rfl"


def test_serialize_marks_bare_name_preceded_by_whitespace():
    # the fixture premise: bare `add_comm` after "theorem " gets marked
    out = serialize_premise("add_comm", "theorem add_comm (a b : α) : a + b = b + a := by ...")
    assert out == "theorem <a>add_comm</a> (a b : α) : a + b = b + a := by ..."


def test_serialize_no_occurrence_leaves_code_unchanged():
    code = "theorem something : x = y := by simp"
    assert serialize_premise("foo.bar", code) == code


# -- retrieval plumbing (fake encoder) --------------------------------------------------------

def _fitted(index_dir: str | None = None, encoder=None):
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = DenseRetriever(index_dir=index_dir, encoder=encoder or FakeEncoder())
    r.build_index(corpus)
    return corpus, r


def test_dense_ranks_exact_serialized_match_first():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    target = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    # a query identical to the premise's serialized text -> cosine 1 -> rank #1
    query = serialize_premise(target.full_name, target.code)
    res = r.retrieve(query, all_uids, k=6)
    assert res[0][0] == target.uid
    assert res[0][1] == pytest.approx(1.0, abs=1e-5)


def test_dense_respects_accessibility():
    corpus, r = _fitted()
    target = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    res = r.retrieve("anything at all", {target.uid}, k=6)
    assert [u for u, _ in res] == [target.uid]


def test_dense_scores_descending_and_capped_at_k():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    res = r.retrieve("a ≤ b", all_uids, k=3)
    scores = [s for _, s in res]
    assert len(res) == 3
    assert scores == sorted(scores, reverse=True)


def test_dense_empty_accessible_returns_empty():
    _corpus, r = _fitted()
    assert r.retrieve("a ≤ b", set(), k=10) == []


def test_dense_persistence_roundtrip(tmp_path):
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    enc1 = FakeEncoder()
    r1 = DenseRetriever(index_dir=str(tmp_path), encoder=enc1)
    r1.build_index(corpus)
    assert (tmp_path / "dense_emb.npy").exists()
    assert (tmp_path / "dense_uids.json").exists()

    # a fresh retriever with a fresh encoder must LOAD the premise vectors (not re-encode them)
    enc2 = CountingEncoder()
    r2 = DenseRetriever(index_dir=str(tmp_path), encoder=enc2)
    r2.build_index(corpus)
    assert enc2.calls == 0                      # premises came from disk, not the encoder

    all_uids = {p.uid for p in corpus.all_premises}
    q = "x : α ⊢ le_refl x"
    assert r1.retrieve(q, all_uids, 6) == r2.retrieve(q, all_uids, 6)


def test_dense_retrieve_before_build_raises():
    r = DenseRetriever(encoder=FakeEncoder())
    with pytest.raises(RuntimeError):
        r.retrieve("q", {"x"}, 5)


def test_dense_index_corruption_detected(tmp_path):
    corpus, r = _fitted(index_dir=str(tmp_path))
    # truncate the uid list so it no longer matches the embedding matrix
    (tmp_path / "dense_uids.json").write_text(json.dumps(["only-one-uid"]), encoding="utf-8")
    r2 = DenseRetriever(index_dir=str(tmp_path), encoder=FakeEncoder())
    with pytest.raises(ValueError):
        r2.build_index(corpus)


# -- premise-text routing (the matched single-vector control uses full_name+code, not ReProver) ---

class RecordingEncoder(FakeEncoder):
    """FakeEncoder that records the exact texts it was asked to encode (to assert which premise
    serialization build_index used)."""

    def __init__(self, dim: int = 48) -> None:
        super().__init__(dim)
        self.seen: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        return super().encode(texts)


def test_premise_text_option_selects_serialization():
    from prooflens.retrievers.bm25 import premise_document

    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    prem = corpus.all_premises

    # matched single-vector control: full_name + code (the LI/pairs text), NOT ReProver's markers
    rec_fnc = RecordingEncoder()
    DenseRetriever(encoder=rec_fnc, premise_text="full_name_code").build_index(corpus)
    assert rec_fnc.seen == [premise_document(p.full_name, p.code) for p in prem]
    assert all("<a>" not in t for t in rec_fnc.seen)

    # default (ReProver) is unchanged -> <a>-marked serialization
    rec_rp = RecordingEncoder()
    DenseRetriever(encoder=rec_rp).build_index(corpus)          # premise_text defaults to reprover
    assert rec_rp.seen == [serialize_premise(p.full_name, p.code) for p in prem]
    assert any("<a>" in t for t in rec_rp.seen)


# -- real ByT5 checkpoint smoke (opt-in; runs on the cluster) ---------------------------------

_HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


@pytest.mark.skipif(
    not (_HAS_TORCH and os.environ.get("PROOFLENS_DENSE_SMOKE")),
    reason="real ByT5 smoke: set PROOFLENS_DENSE_SMOKE=1 with torch+transformers (+ staged model)",
)
def test_reprover_encoder_real_smoke():
    from prooflens.retrievers.dense import _ByT5Encoder

    model_path = os.environ.get(
        "PROOFLENS_DENSE_MODEL", "kaiyuy/leandojo-lean4-retriever-byt5-small"
    )
    enc = _ByT5Encoder(model_path, max_length=1024, batch_size=8, device="cpu")
    embs = enc.encode(["a b : α ⊢ a + b = b + a", "theorem <a>add_comm</a> (a b) : ..."])
    assert embs.shape == (2, enc.dim)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)  # unit-norm (F.normalize)
    two = enc.encode(["same text", "same text"])
    assert float(two[0] @ two[1]) == pytest.approx(1.0, abs=1e-4)     # identical text -> cosine 1


@pytest.mark.skipif(
    not (importlib.util.find_spec("sentence_transformers") and os.environ.get("PROOFLENS_SV_SMOKE")
         and os.environ.get("MODELS_DIR")),
    reason="single-vector encoder smoke: set PROOFLENS_SV_SMOKE=1 with sentence-transformers",
)
def test_st_encoder_real_smoke():
    from prooflens.retrievers.dense import _STEncoder

    path = str(Path(os.environ["MODELS_DIR"]) / "Alibaba-NLP__gte-modernbert-base")
    enc = _STEncoder(path, max_length=128, batch_size=4, device="cpu")
    embs = enc.encode(["a b : α ⊢ a + b = b + a", "add_comm : a + b = b + a"])
    assert embs.shape == (2, enc.dim)
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-4)  # normalize_embeddings=True
    two = enc.encode(["same text", "same text"])
    assert float(two[0] @ two[1]) == pytest.approx(1.0, abs=1e-4)
