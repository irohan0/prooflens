"""Hermetic tests for the pure token-IDF logic (no torch, no tokenizer)."""

from __future__ import annotations

import math

import pytest

from prooflens.retrievers.token_idf import (
    TokenIDF,
    document_frequencies,
    idf_from_df,
)

# -- document frequency ---------------------------------------------------------------------------

def test_document_frequency_counts_documents_not_occurrences():
    docs = [
        ["add", "comm", "add"],     # 'add' twice in ONE doc -> df 1, not 2
        ["add", "mul"],
        ["mul"],
    ]
    df = document_frequencies(docs)
    assert df["add"] == 2
    assert df["mul"] == 2
    assert df["comm"] == 1


def test_document_frequency_dedupes_per_document():
    # even passed as a list with repeats, each doc contributes at most 1 to a token's df
    df = document_frequencies([["x", "x", "x"], ["x"]])
    assert df["x"] == 2


# -- the IDF formula ------------------------------------------------------------------------------

def test_idf_formula_matches_smooth_idf():
    # idf(t) = log((N+1)/(df+1)) + 1
    n = 100
    df = {"rare": 1, "common": 100, "mid": 10}
    idf = idf_from_df(df, n)
    assert idf["rare"] == pytest.approx(math.log(101 / 2) + 1)
    assert idf["common"] == pytest.approx(math.log(101 / 101) + 1)   # == 1.0 exactly
    assert idf["mid"] == pytest.approx(math.log(101 / 11) + 1)


def test_idf_is_monotone_rarer_gets_more_weight():
    idf = idf_from_df({"a": 1, "b": 5, "c": 50, "d": 500}, 500)
    assert idf["a"] > idf["b"] > idf["c"] > idf["d"]


def test_token_in_every_document_gets_weight_one_not_zero():
    # a token in ALL docs must not be zeroed out (still contributes its raw MaxSim, just no boost)
    idf = idf_from_df({"everywhere": 20}, 20)
    assert idf["everywhere"] == pytest.approx(1.0)


def test_idf_rejects_negative_n_docs():
    with pytest.raises(ValueError, match="n_docs"):
        idf_from_df({"a": 1}, -1)


# -- the TokenIDF table ---------------------------------------------------------------------------

def test_from_document_tokens_end_to_end():
    docs = [["Nat", "add", "comm"], ["Nat", "mul"], ["mul"]]
    table = TokenIDF.from_document_tokens(docs)
    assert table.n_docs == 3
    # 'comm' (df 1) is rarer than 'Nat' (df 2) is rarer than 'mul' (df 2) ... check ordering
    assert table.weight("comm") > table.weight("Nat")


def test_weight_fallback_for_unknown_tokens():
    table = TokenIDF.from_document_tokens([["a"], ["a", "b"]])
    assert table.weight("a") == pytest.approx(math.log(3 / 3) + 1)   # df 2 of 2 -> 1.0
    assert table.weight("[MASK]") == 1.0                             # unknown -> default fallback
    assert table.weight("[MASK]", fallback=0.0) == 0.0               # honours the fallback value


def test_save_and_load_roundtrip(tmp_path):
    docs = [["Nat", "add"], ["mul", "add"], ["Nat"]]
    table = TokenIDF.from_document_tokens(docs, meta={"source": "unit test"})
    p = tmp_path / "token_idf.json"
    table.save(p)
    loaded = TokenIDF.load(p)
    assert loaded.n_docs == table.n_docs
    assert loaded.meta["source"] == "unit test"
    for t in ("Nat", "add", "mul"):
        assert loaded.weight(t) == pytest.approx(table.weight(t))


def test_len_reports_distinct_token_count():
    table = TokenIDF.from_document_tokens([["a", "b"], ["b", "c"]])
    assert len(table) == 3
