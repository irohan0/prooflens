"""Unit tests for the late-interaction (ColBERT/MaxSim) retriever.

Hermetic (numpy only): a deterministic fake ColBERT encoder returns ragged per-text token
embeddings, so the whole path — exact MaxSim over the accessible set (the vectorised `reduceat`
segment-max), accessibility masking, persistence roundtrip, the weighting seam — is exercised
without PyLate/torch/GPU. The vectorised score is checked against the naive `maxsim_score`
reference. A real PyLate smoke is opt-in (`PROOFLENS_LI_SMOKE=1` + pylate importable).
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
from prooflens.retrievers.bm25 import premise_document
from prooflens.retrievers.late_interaction import (
    LateInteractionRetriever,
    is_symbol_subword,
    maxsim_score,
)

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


# -- deterministic fake ColBERT encoder (ragged token embeddings) -----------------------------

class FakeColBERT:
    """Whitespace-tokenizes text; maps each word to a fixed unit vector (sha1-seeded, stable
    across processes). Shared words between a query and a premise yield MaxSim contributions of 1,
    so an exact-text query ranks its premise first — enough to test all retrieval plumbing."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _vec(self, word: str) -> np.ndarray:
        seed = int(hashlib.sha1(word.encode("utf-8")).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / (float(np.linalg.norm(v)) + 1e-12)

    def _tokens(self, text: str) -> np.ndarray:
        words = text.split() or ["<empty>"]
        return np.stack([self._vec(w) for w in words]).astype(np.float32)

    special_tokens = frozenset()

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._tokens(t) for t in texts]

    def encode_queries(self, texts: list[str]) -> list[np.ndarray]:
        return [self._tokens(t) for t in texts]

    def encode_query_with_tokens(self, text: str) -> tuple[np.ndarray, list[str]]:
        # embeddings identical to encode_queries (one row per whitespace word), aligned with words
        words = text.split() or ["<empty>"]
        return self._tokens(text), words


class CountingColBERT(FakeColBERT):
    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim)
        self.doc_calls = 0

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.doc_calls += 1
        return super().encode_documents(texts)


# -- MaxSim reference --------------------------------------------------------------------------

def test_maxsim_score_hand_computed():
    # 2 query tokens, 2 doc tokens, all unit vectors along axes -> cosines are 0/1
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    # q0 best-matches d (cos 1); q1 orthogonal to both d tokens (cos 0) -> score 1.0
    assert maxsim_score(q, d) == pytest.approx(1.0)
    # with weights [2, 3]: 2*1 + 3*0 = 2.0
    assert maxsim_score(q, d, np.array([2.0, 3.0], dtype=np.float32)) == pytest.approx(2.0)


# -- retrieval plumbing (fake encoder) --------------------------------------------------------

def _fitted(index_dir: str | None = None, encoder=None, weighting=None):
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = LateInteractionRetriever(
        index_dir=index_dir, encoder=encoder or FakeColBERT(), weighting=weighting
    )
    r.build_index(corpus)
    return corpus, r


def test_li_vectorised_matches_naive_maxsim():
    corpus, r = _fitted()
    enc = FakeColBERT()
    all_uids = {p.uid for p in corpus.all_premises}
    query = "x : α ⊢ le_refl x le_trans"
    e_q = enc.encode_queries([query])[0]
    naive = {
        p.uid: maxsim_score(e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0])
        for p in corpus.all_premises
    }
    res = r.retrieve(query, all_uids, k=6)
    # every returned score matches the independent naive MaxSim for that premise (correct pairing);
    for uid, score in res:                              # vectorised reduceat == naive definition
        assert score == pytest.approx(naive[uid], abs=1e-4)
    assert {u for u, _ in res} == set(naive)            # all premises returned (k >= corpus size)
    # ...and the result is internally sorted by its OWN scores (score desc, uid tie-break). We do
    # not compare the exact order to a *separate* naive computation: the two paths differ by ~1 ULP,
    # which flips genuinely-tied premises (add_comm/mul_comm here) — a test artefact, not a bug.
    assert res == sorted(res, key=lambda t: (-t[1], t[0]))


def test_li_ranks_exact_text_match_first():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    target = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    query = premise_document(target.full_name, target.code)   # identical text -> all tokens match
    res = r.retrieve(query, all_uids, k=6)
    assert res[0][0] == target.uid


def test_li_respects_accessibility():
    corpus, r = _fitted()
    target = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    res = r.retrieve("anything here", {target.uid}, k=6)
    assert [u for u, _ in res] == [target.uid]


def test_li_empty_accessible_returns_empty():
    _corpus, r = _fitted()
    assert r.retrieve("a ≤ b", set(), k=10) == []


def test_li_scores_descending_and_capped_at_k():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    res = r.retrieve("a ≤ b → c", all_uids, k=3)
    scores = [s for _, s in res]
    assert len(res) == 3
    assert scores == sorted(scores, reverse=True)


def _assert_consistent_ranking(a, b, tol: float = 1e-6) -> None:
    """Same premises retrieved with matching scores. Exact ordered-list equality is unsafe here:
    near-tied low scores can reorder across independent BLAS matmuls (~1 ULP), so the real invariant
    is identical uid set + per-uid scores within tolerance (not the exact order)."""
    da, db = dict(a), dict(b)
    assert da.keys() == db.keys()
    for u in da:
        assert da[u] == pytest.approx(db[u], abs=tol)


def test_li_accessible_cache_reuse_is_consistent():
    corpus, r = _fitted()
    acc = {p.uid for p in corpus.all_premises}
    q = "le_trans a b c"
    first = r.retrieve(q, acc, 6)          # builds the gather cache
    second = r.retrieve(q, acc, 6)         # same object -> cache hit
    third = r.retrieve(q, {u for u in acc}, 6)   # fresh equal object -> cache rebuild
    _assert_consistent_ranking(first, second)
    _assert_consistent_ranking(first, third)


def test_li_persistence_roundtrip(tmp_path):
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r1 = LateInteractionRetriever(index_dir=str(tmp_path), encoder=FakeColBERT())
    r1.build_index(corpus)
    assert (tmp_path / "li_tok.npy").exists()
    assert (tmp_path / "li_off.npy").exists()
    assert (tmp_path / "li_uids.json").exists()

    enc2 = CountingColBERT()
    r2 = LateInteractionRetriever(index_dir=str(tmp_path), encoder=enc2)
    r2.build_index(corpus)
    assert enc2.doc_calls == 0                         # premise tokens loaded, not re-encoded

    all_uids = {p.uid for p in corpus.all_premises}
    q = "x : α ⊢ le_refl x"
    _assert_consistent_ranking(r1.retrieve(q, all_uids, 6), r2.retrieve(q, all_uids, 6))


def test_li_index_corruption_detected(tmp_path):
    corpus, _r = _fitted(index_dir=str(tmp_path))
    (tmp_path / "li_uids.json").write_text(json.dumps(["only-one"]), encoding="utf-8")
    r2 = LateInteractionRetriever(index_dir=str(tmp_path), encoder=FakeColBERT())
    with pytest.raises(ValueError):
        r2.build_index(corpus)


def test_li_retrieve_before_build_raises():
    r = LateInteractionRetriever(encoder=FakeColBERT())
    with pytest.raises(RuntimeError):
        r.retrieve("q", {"x"}, 5)


# -- symbol-anchored token weighting (Phase 9) ------------------------------------------------

def test_is_symbol_subword_classification():
    specials = frozenset({"[CLS]", "[SEP]", "[MASK]", "[Q]", "[D]", "[PAD]"})
    # identifier / name fragments (with ModernBERT "Ġ" space marker) -> symbol
    assert is_symbol_subword("Ġadd", specials)
    assert is_symbol_subword("comm", specials)
    assert is_symbol_subword("le_refl", specials)
    assert is_symbol_subword("Ġ123", specials)
    # mathematical operators -> symbol
    assert is_symbol_subword("Ġ≤", specials)
    assert is_symbol_subword("→", specials)
    assert is_symbol_subword("Ġ∀", specials)
    # special tokens, whitespace, pure syntactic filler -> NOT symbol
    assert not is_symbol_subword("[MASK]", specials)
    assert not is_symbol_subword("[Q]", specials)
    assert not is_symbol_subword("Ġ", specials)            # pure space marker
    assert not is_symbol_subword("Ġ⊢", specials)           # turnstile (structural)
    assert not is_symbol_subword("Ġ:", specials)           # type-ascription colon
    assert not is_symbol_subword("(", specials)
    assert not is_symbol_subword(",", specials)


def test_li_weighting_on_scores_match_weighted_maxsim():
    weighting = {"enabled": True, "symbol_weight": 3.0, "default_weight": 1.0}
    corpus, r = _fitted(weighting=weighting)
    enc = FakeColBERT()
    all_uids = {p.uid for p in corpus.all_premises}
    query = "le_trans ≤ a ⊢ b"
    e_q, tokens = enc.encode_query_with_tokens(query)
    weights = np.array(
        [3.0 if is_symbol_subword(t, enc.special_tokens) else 1.0 for t in tokens], dtype=np.float32
    )
    naive = {
        p.uid: maxsim_score(
            e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0], weights
        )
        for p in corpus.all_premises
    }
    res = r.retrieve(query, all_uids, k=6)
    for uid, score in res:                                 # ON path applies the weights correctly
        assert score == pytest.approx(naive[uid], abs=1e-4)
    assert res == sorted(res, key=lambda t: (-t[1], t[0]))


def test_li_weighting_on_changes_scores_vs_off():
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    off = LateInteractionRetriever(encoder=FakeColBERT(), weighting={"enabled": False})
    on = LateInteractionRetriever(
        encoder=FakeColBERT(),
        weighting={"enabled": True, "symbol_weight": 5.0, "default_weight": 1.0},
    )
    off.build_index(corpus)
    on.build_index(corpus)
    all_uids = {p.uid for p in corpus.all_premises}
    q = "le_trans ≤ a ⊢ b"                                 # mixes symbol + filler tokens
    off_scores = [s for _, s in off.retrieve(q, all_uids, 6)]
    on_scores = [s for _, s in on.retrieve(q, all_uids, 6)]
    assert off_scores != on_scores                         # weighting actually changes the score


def test_li_weighting_off_is_unweighted_maxsim():
    # OFF must equal plain MaxSim (weights all one) — guards the ablation's OFF arm.
    corpus, r = _fitted(weighting={"enabled": False})
    enc = FakeColBERT()
    all_uids = {p.uid for p in corpus.all_premises}
    q = "le_trans a b"
    e_q = enc.encode_queries([q])[0]
    naive = {
        p.uid: maxsim_score(e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0])
        for p in corpus.all_premises
    }
    for uid, score in r.retrieve(q, all_uids, k=6):
        assert score == pytest.approx(naive[uid], abs=1e-4)


# -- Phase 22: IDF-weighted modes -------------------------------------------------------------

def _idf_retriever(mode, encoder, idf_table, **extra):
    """A fitted LI retriever in an IDF weighting mode, with the IDF table injected (no file)."""
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = LateInteractionRetriever(
        encoder=encoder,
        weighting={"enabled": True, "mode": mode, "idf_fallback": 1.0, **extra},
    )
    r._idf = idf_table                       # inject (bypasses file loading)
    r.build_index(corpus)
    return corpus, r


def test_idf_mode_weights_each_token_by_its_idf():
    from prooflens.retrievers.token_idf import TokenIDF

    # FakeColBERT tokenizes on whitespace, so IDF keys are whole words.
    idf = TokenIDF({"le_trans": 5.0, "a": 1.0, "b": 1.0}, n_docs=3)
    enc = FakeColBERT()
    corpus, r = _idf_retriever("idf", enc, idf)
    all_uids = {p.uid for p in corpus.all_premises}
    q = "le_trans a b"
    e_q, tokens = enc.encode_query_with_tokens(q)
    weights = np.array([idf.weight(t, 1.0) for t in tokens], dtype=np.float32)  # [5,1,1]
    naive = {
        p.uid: maxsim_score(
            e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0], weights
        )
        for p in corpus.all_premises
    }
    for uid, score in r.retrieve(q, all_uids, k=6):
        assert score == pytest.approx(naive[uid], abs=1e-4)


def test_idf_mode_unknown_token_uses_fallback():
    from prooflens.retrievers.token_idf import TokenIDF

    idf = TokenIDF({"known": 9.0}, n_docs=2)     # 'mystery' is absent -> fallback
    enc = FakeColBERT()
    corpus, r = _idf_retriever("idf", enc, idf, idf_fallback=0.5)
    all_uids = {p.uid for p in corpus.all_premises}
    q = "known mystery"
    e_q, tokens = enc.encode_query_with_tokens(q)
    weights = np.array([9.0 if t == "known" else 0.5 for t in tokens], dtype=np.float32)
    naive = {
        p.uid: maxsim_score(
            e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0], weights
        )
        for p in corpus.all_premises
    }
    for uid, score in r.retrieve(q, all_uids, k=6):
        assert score == pytest.approx(naive[uid], abs=1e-4)


def test_symbol_idf_mode_blends_idf_on_symbols_with_default_on_filler():
    from prooflens.retrievers.token_idf import TokenIDF

    idf = TokenIDF({"le_trans": 7.0, "≤": 3.0}, n_docs=4)
    enc = FakeColBERT()
    corpus, r = _idf_retriever("symbol_idf", enc, idf, default_weight=1.0)
    all_uids = {p.uid for p in corpus.all_premises}
    q = "le_trans ≤ ⊢"          # le_trans + ≤ are symbols; ⊢ is filler
    e_q, tokens = enc.encode_query_with_tokens(q)
    weights = np.array(
        [idf.weight(t, 1.0) if is_symbol_subword(t, enc.special_tokens) else 1.0 for t in tokens],
        dtype=np.float32,
    )
    naive = {
        p.uid: maxsim_score(
            e_q, enc.encode_documents([premise_document(p.full_name, p.code)])[0], weights
        )
        for p in corpus.all_premises
    }
    for uid, score in r.retrieve(q, all_uids, k=6):
        assert score == pytest.approx(naive[uid], abs=1e-4)


def test_idf_scale_multiplies_symbol_weights_in_blend():
    from prooflens.retrievers.token_idf import TokenIDF

    idf = TokenIDF({"le_trans": 2.0}, n_docs=4)
    enc = FakeColBERT()
    weights_at = {}
    for scale in (1.0, 3.0):
        corpus, r = _idf_retriever("symbol_idf", enc, idf, idf_scale=scale)
        w = r._token_weights(["le_trans", "⊢"], enc)
        weights_at[scale] = w
    # symbol 'le_trans' weight scales 2.0 -> 6.0; filler '⊢' stays at default 1.0
    assert weights_at[1.0][0] == pytest.approx(2.0)
    assert weights_at[3.0][0] == pytest.approx(6.0)
    assert weights_at[1.0][1] == pytest.approx(1.0)
    assert weights_at[3.0][1] == pytest.approx(1.0)


def test_idf_scaling_all_tokens_does_not_change_ranking():
    # Pure-idf mode is scale-invariant: multiplying every weight by c reorders nothing.
    from prooflens.retrievers.token_idf import TokenIDF

    idf = TokenIDF({"le_trans": 5.0, "le_refl": 2.0, "a": 1.0}, n_docs=3)
    enc = FakeColBERT()
    q = "le_trans le_refl a"
    corpus1, r1 = _idf_retriever("idf", enc, idf, idf_scale=1.0)
    corpus2, r2 = _idf_retriever("idf", enc, idf, idf_scale=10.0)
    uids = {p.uid for p in corpus1.all_premises}
    order1 = [u for u, _ in r1.retrieve(q, uids, k=6)]
    order2 = [u for u, _ in r2.retrieve(q, uids, k=6)]
    assert order1 == order2


def test_unknown_weighting_mode_raises():
    from prooflens.retrievers.token_idf import TokenIDF

    enc = FakeColBERT()
    corpus, r = _idf_retriever("nonsense", enc, TokenIDF({"a": 1.0}, 1))
    uids = {p.uid for p in corpus.all_premises}     # real accessible set -> weighting path runs
    with pytest.raises(ValueError, match="unknown weighting mode"):
        r.retrieve("le_trans a", uids, k=1)


def test_idf_mode_without_table_or_path_raises():
    # Missing table + no index_dir/path -> a clear error, not a silent wrong weighting.
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = LateInteractionRetriever(
        encoder=FakeColBERT(), weighting={"enabled": True, "mode": "idf"}
    )
    r.build_index(corpus)
    uids = {p.uid for p in corpus.all_premises}
    with pytest.raises(RuntimeError, match="idf weighting needs"):
        r.retrieve("le_trans a", uids, k=1)


def test_idf_table_loads_from_index_dir(tmp_path):
    from prooflens.retrievers.token_idf import TokenIDF

    TokenIDF({"le_trans": 4.0}, n_docs=2).save(tmp_path / "token_idf.json")
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = LateInteractionRetriever(
        index_dir=str(tmp_path), encoder=FakeColBERT(),
        weighting={"enabled": True, "mode": "idf"},
    )
    r.build_index(corpus)
    uids = {p.uid for p in corpus.all_premises}
    res = r.retrieve("le_trans a", uids, k=3)      # loads token_idf.json from index_dir, no error
    assert res and r._idf is not None and r._idf.weight("le_trans") == pytest.approx(4.0)


# -- real PyLate ColBERT smoke (opt-in; runs on the cluster) ----------------------------------

_HAS_PYLATE = importlib.util.find_spec("pylate") is not None


@pytest.mark.skipif(
    not (_HAS_PYLATE and os.environ.get("PROOFLENS_LI_SMOKE")),
    reason="real ColBERT smoke: set PROOFLENS_LI_SMOKE=1 with pylate installed (+ staged model)",
)
def test_colbert_encoder_real_smoke():
    from prooflens.retrievers.late_interaction import _ColBERTEncoder

    model_path = os.environ.get("PROOFLENS_LI_MODEL", "lightonai/GTE-ModernColBERT-v1")
    # pin CPU: this is a login-node pre-flight; the login node's GPU is often busy/unavailable.
    enc = _ColBERTEncoder(model_path, query_length=64, document_length=128, batch_size=4,
                          device="cpu")
    docs = enc.encode_documents(["theorem add_comm (a b) : a + b = b + a"])
    qs = enc.encode_queries(["a b : α ⊢ a + b = b + a"])
    assert docs[0].ndim == 2 and qs[0].ndim == 2
    assert docs[0].shape[1] == qs[0].shape[1]                       # same token dim (128)
    assert np.allclose(np.linalg.norm(docs[0], axis=1), 1.0, atol=1e-3)  # unit-norm tokens
    # a query equal to the doc text should out-score an unrelated doc
    same = enc.encode_queries(["theorem add_comm (a b) : a + b = b + a"])[0]
    other = enc.encode_documents(["lemma foo : True"])[0]
    assert maxsim_score(same, docs[0]) > maxsim_score(same, other)

    # Phase 9 alignment: the token-aligned query encode must be 1:1 with the sub-word strings and
    # produce the SAME vectors as encode_queries, so ON vs OFF differs only by the weights.
    state = "a b : α ⊢ a + b = b + a"
    emb_off = enc.encode_queries([state])[0]
    emb_on, tokens = enc.encode_query_with_tokens(state)
    assert emb_on.shape[0] == len(tokens)                          # 1:1 token alignment
    assert emb_on.shape == emb_off.shape                           # same tokenization as OFF
    assert np.allclose(emb_on, emb_off, atol=1e-4)                 # same vectors -> clean ablation
    tags = [is_symbol_subword(t, enc.special_tokens) for t in tokens]
    assert any(tags) and not all(tags)                             # some symbol, some filler
