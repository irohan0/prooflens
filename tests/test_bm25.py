"""Unit tests for the BM25 retriever and its Lean-aware tokenizer."""

from __future__ import annotations

from pathlib import Path

from prooflens.data.corpus import load_corpus
from prooflens.retrievers.bm25 import BM25Retriever, lean_tokenize, premise_document

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


# -- tokenizer --------------------------------------------------------------------------------

def test_tokenizer_splits_dotted_names_and_keeps_identifiers():
    assert lean_tokenize("Nat.add_comm a b") == ["Nat", ".", "add_comm", "a", "b"]


def test_tokenizer_keeps_operators_and_unicode_as_tokens():
    assert lean_tokenize("a ≤ b → c") == ["a", "≤", "b", "→", "c"]


def test_tokenizer_preserves_primes_and_case():
    assert lean_tokenize("le_refl'") == ["le_refl'"]
    assert lean_tokenize("Continuous") == ["Continuous"]        # not lowercased by default


def test_tokenizer_lowercase_flag():
    assert lean_tokenize("Nat", lowercase=True) == ["nat"]


def test_premise_document_prepends_full_name():
    assert premise_document("Nat.add", "theorem Nat.add ...").startswith("Nat.add theorem")


# -- ranking ----------------------------------------------------------------------------------

def _fitted():
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r = BM25Retriever()
    r.build_index(corpus)
    return corpus, r


def test_bm25_ranks_the_named_premise_first():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    cont = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    res = r.retrieve("⊢ continuous_id applied here", all_uids, k=6)
    assert res[0][0] == cont.uid            # unique high-IDF name token dominates
    assert res[0][1] > 0.0


def test_bm25_respects_accessibility():
    corpus, r = _fitted()
    cont = next(p for p in corpus.all_premises if p.full_name == "continuous_id")
    # only continuous_id is accessible -> nothing else may be returned
    res = r.retrieve("⊢ continuous_id applied here", {cont.uid}, k=6)
    assert [u for u, _ in res] == [cont.uid]


def test_bm25_scores_descending_and_capped_at_k():
    corpus, r = _fitted()
    all_uids = {p.uid for p in corpus.all_premises}
    res = r.retrieve("a ≤ b", all_uids, k=3)
    scores = [s for _, s in res]
    assert len(res) == 3
    assert scores == sorted(scores, reverse=True)


def test_bm25_persistence_roundtrip(tmp_path):
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    r1 = BM25Retriever(index_dir=str(tmp_path))
    r1.build_index(corpus)
    assert (tmp_path / "bm25.pkl").exists()
    # a fresh retriever loads the persisted index and ranks identically
    r2 = BM25Retriever(index_dir=str(tmp_path))
    r2.build_index(corpus)          # should load, not rebuild
    all_uids = {p.uid for p in corpus.all_premises}
    q = "x : α ⊢ le_refl x"
    assert r1.retrieve(q, all_uids, 6) == r2.retrieve(q, all_uids, 6)
