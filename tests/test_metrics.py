"""Unit tests for eval/metrics.py with HAND-COMPUTED expected values.

Every expected number below is worked out by hand in a comment so a human can verify the math
without trusting the implementation. Covers the edge cases from docs/EVALUATION.md §2: empty gold,
no gold retrieved, fewer than k accessible, duplicate premise names, and tie/ordering behaviour.
If the metrics are wrong, every downstream number is wrong — so these are the project's floor.

log2(2) = 1, log2(3) = 1.5849625007, log2(4) = 2.
"""

import math

import pytest

from prooflens.eval.metrics import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Discount at rank r is 1 / log2(r + 1). Handy constants for the hand computations.
D1 = 1.0 / math.log2(2)  # rank 1 -> 1.0
D2 = 1.0 / math.log2(3)  # rank 2 -> 0.6309297536
D3 = 1.0 / math.log2(4)  # rank 3 -> 0.5


# --------------------------------------------------------------------------------------------
# Recall@k
# --------------------------------------------------------------------------------------------

def test_recall_basic():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "c"}
    # top-1 = {a}; hit {a}; 1/|gold|=1/2
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0.5)
    # top-10 = all; hits {a,c}=2; 2/2 = 1.0
    assert recall_at_k(ranked, gold, 10) == pytest.approx(1.0)


def test_recall_no_gold_retrieved():
    # gold present in corpus but never retrieved -> 0
    assert recall_at_k(["x", "y", "z"], {"a"}, 10) == 0.0


def test_recall_fewer_than_k_available():
    # only 1 premise retrieved, k=10; hit a; |gold|=2 -> 1/2
    assert recall_at_k(["a"], {"a", "b"}, 10) == pytest.approx(0.5)


def test_recall_dedupes_duplicates():
    # "a" duplicated must not count twice: dedupe -> ["a"]; 1/|gold|=1/1
    assert recall_at_k(["a", "a"], {"a"}, 10) == pytest.approx(1.0)


def test_recall_capped_by_k_when_gold_larger():
    # top-1 can hold at most one of two gold items -> 1/2, never > that
    assert recall_at_k(["a", "b"], {"a", "b"}, 1) == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# Reciprocal rank / MRR (per-example)
# --------------------------------------------------------------------------------------------

def test_rr_first_position():
    assert reciprocal_rank(["a", "b", "c"], {"a", "c"}) == pytest.approx(1.0)


def test_rr_second_position():
    # first gold ("a") is at rank 2 -> 1/2
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_rr_none_retrieved():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_rr_dedupe_improves_rank():
    # ["a","a","b"] -> dedupe ["a","b"]; first gold "b" at rank 2 -> 1/2 (not 1/3)
    assert reciprocal_rank(["a", "a", "b"], {"b"}) == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# nDCG@k
# --------------------------------------------------------------------------------------------

def test_ndcg_basic_at_2():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "c"}
    # top-2 = [a, b]; DCG = D1 (a hit at rank1) = 1.0
    # ideal: 2 gold in 2 slots -> IDCG = D1 + D2 = 1 + 0.6309297536 = 1.6309297536
    # nDCG = 1.0 / 1.6309297536 = 0.6131471928
    expected = D1 / (D1 + D2)
    assert ndcg_at_k(ranked, gold, 2) == pytest.approx(expected)
    assert ndcg_at_k(ranked, gold, 2) == pytest.approx(0.6131471928, rel=1e-6)


def test_ndcg_gold_lower_ranked():
    ranked = ["x", "a", "b"]
    gold = {"a", "b"}
    # top-3 = [x, a, b]; DCG = D2 (a@2) + D3 (b@3) = 0.6309297536 + 0.5 = 1.1309297536
    # ideal: 2 gold -> IDCG = D1 + D2 = 1.6309297536
    # nDCG = 1.1309297536 / 1.6309297536 = 0.6934264
    expected = (D2 + D3) / (D1 + D2)
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(expected)
    assert ndcg_at_k(ranked, gold, 3) == pytest.approx(0.6934264, rel=1e-6)


def test_ndcg_none_retrieved():
    assert ndcg_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_ndcg_perfect_ordering_is_one():
    # both gold at the very top -> DCG == IDCG -> 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_ndcg_fewer_than_k_available():
    ranked = ["a"]
    gold = {"a", "b"}
    # top-10 -> [a]; DCG = D1 = 1.0; ideal 2 gold -> IDCG = D1 + D2 = 1.6309297536
    expected = D1 / (D1 + D2)
    assert ndcg_at_k(ranked, gold, 10) == pytest.approx(expected)


def test_ndcg_dedupe_duplicates():
    # ["b","b","a"] -> dedupe ["b","a"]; both gold; DCG = D1 + D2 == IDCG -> 1.0
    assert ndcg_at_k(["b", "b", "a"], {"a", "b"}, 10) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------------
# Average precision (MAP per-example) — optional extra metric
# --------------------------------------------------------------------------------------------

def test_ap_basic():
    # ["a","x","b"], gold {a,b}: a@1 -> 1/1=1.0; b@3 -> 2/3=0.6667; sum/|gold| = 1.6667/2
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx((1.0 + 2 / 3) / 2)
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(0.8333333, rel=1e-6)


def test_ap_perfect_is_one():
    # both gold at the top: 1/1 + 2/2 = 2.0; /2 = 1.0
    assert average_precision(["a", "b", "c"], {"a", "b"}) == pytest.approx(1.0)


def test_ap_none_retrieved():
    assert average_precision(["x", "y"], {"a"}) == 0.0


def test_ap_dedupes():
    # ["a","a","b"] -> ["a","b"]; 1/1 + 2/2 = 2.0; /2 = 1.0
    assert average_precision(["a", "a", "b"], {"a", "b"}) == pytest.approx(1.0)


def test_ap_at_k_truncates():
    # k=2 keeps ["a","x"]; only a@1 -> 1.0; /|gold| 2 = 0.5
    assert average_precision(["a", "x", "b"], {"a", "b"}, k=2) == pytest.approx(0.5)


def test_ap_empty_gold_raises():
    with pytest.raises(ValueError):
        average_precision(["a"], set())


# --------------------------------------------------------------------------------------------
# Empty gold -> loud failure (documented decision: not a valid example)
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda: recall_at_k(["a"], set(), 10),
        lambda: reciprocal_rank(["a"], set()),
        lambda: ndcg_at_k(["a"], set(), 10),
    ],
)
def test_empty_gold_raises(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(k):
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], {"a"}, k)
