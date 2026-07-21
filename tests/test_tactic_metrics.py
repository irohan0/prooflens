"""Hermetic, hand-computed tests for the generated-tactic metrics."""

from __future__ import annotations

import pytest

from prooflens.eval.tactic_metrics import (
    first_match_rank,
    normalize_tactic,
    premise_name_match_at_k,
    tactic_match_at_k,
)

# -- normalization --------------------------------------------------------------------------------

def test_normalize_collapses_whitespace_runs_and_strips():
    assert normalize_tactic("  simp   only \n [foo]  ") == "simp only [foo]"


def test_normalize_treats_newline_and_space_as_equal():
    assert normalize_tactic("rw [a,\n b]") == normalize_tactic("rw [a, b]")


def test_normalize_does_not_equate_semantically_different_tactics():
    # Guard against over-normalising: `simp` and `simp only` must stay distinct, as must
    # argument order. Making these equal would inflate the score.
    assert normalize_tactic("simp [foo]") != normalize_tactic("simp only [foo]")
    assert normalize_tactic("rw [a, b]") != normalize_tactic("rw [b, a]")


# -- match@k --------------------------------------------------------------------------------------

def test_top1_match_hit_and_miss():
    assert tactic_match_at_k(["exact add_comm a b"], "exact add_comm a b", 1) == 1.0
    assert tactic_match_at_k(["exact mul_comm a b"], "exact add_comm a b", 1) == 0.0


def test_match_at_k_finds_a_later_candidate_but_top1_does_not():
    cands = ["simp", "ring", "exact add_comm a b"]
    assert tactic_match_at_k(cands, "exact add_comm a b", 1) == 0.0
    assert tactic_match_at_k(cands, "exact add_comm a b", 2) == 0.0
    assert tactic_match_at_k(cands, "exact add_comm a b", 3) == 1.0


def test_k_larger_than_candidate_list_is_fine():
    assert tactic_match_at_k(["ring"], "ring", 10) == 1.0


def test_normalization_toggle_changes_the_verdict():
    cands = ["rw [a,\n  b]"]
    assert tactic_match_at_k(cands, "rw [a, b]", 1, normalize=True) == 1.0
    assert tactic_match_at_k(cands, "rw [a, b]", 1, normalize=False) == 0.0


def test_duplicates_do_not_occupy_two_slots():
    # "simp" repeated must not push the correct tactic out of the top-2 window.
    cands = ["simp", "simp", "ring"]
    assert tactic_match_at_k(cands, "ring", 2) == 1.0


def test_empty_reference_raises():
    with pytest.raises(ValueError, match="empty reference"):
        tactic_match_at_k(["simp"], "", 1)


def test_bad_k_raises():
    with pytest.raises(ValueError, match="k must be >= 1"):
        tactic_match_at_k(["simp"], "simp", 0)


def test_no_candidates_scores_zero():
    assert tactic_match_at_k([], "ring", 1) == 0.0


# -- first_match_rank -----------------------------------------------------------------------------

def test_first_match_rank_is_one_indexed():
    assert first_match_rank(["simp", "ring"], "ring") == 2
    assert first_match_rank(["ring", "simp"], "ring") == 1


def test_first_match_rank_none_when_absent():
    assert first_match_rank(["simp", "ring"], "omega") is None


def test_first_match_rank_uses_deduped_positions():
    # Deduping keeps "ring" at rank 2, not 3, so a persisted rank stays consistent with match@k.
    assert first_match_rank(["simp", "simp", "ring"], "ring") == 2
    assert tactic_match_at_k(["simp", "simp", "ring"], "ring", 2) == 1.0


# -- premise_name@k -------------------------------------------------------------------------------

def test_premise_name_credits_the_real_pilot_case():
    """The case that motivated this metric: right lemma, wrong location specifier."""
    reference = "rw [mem_skewAdjointSubmodule] at *"
    generated = ["rw [mem_skewAdjointSubmodule] at hf hg"]
    gold = {"mem_skewAdjointSubmodule"}
    assert tactic_match_at_k(generated, reference, 1) == 0.0        # exact match says "wrong"
    assert premise_name_match_at_k(generated, gold, 1) == 1.0       # but the lemma IS right


def test_premise_name_requires_a_whole_token_not_a_substring():
    # `add_comm` must NOT be credited by `add_comm_sub` — substring matching would inflate this.
    assert premise_name_match_at_k(["exact add_comm_sub x"], {"add_comm"}, 1) == 0.0
    assert premise_name_match_at_k(["exact add_comm x"], {"add_comm"}, 1) == 1.0


def test_premise_name_miss_when_no_gold_name_appears():
    assert premise_name_match_at_k(["simp", "ring"], {"add_comm"}, 2) == 0.0


def test_premise_name_any_gold_name_counts():
    assert premise_name_match_at_k(["exact mul_comm a b"], {"add_comm", "mul_comm"}, 1) == 1.0


def test_premise_name_respects_k():
    cands = ["simp", "exact add_comm a b"]
    assert premise_name_match_at_k(cands, {"add_comm"}, 1) == 0.0
    assert premise_name_match_at_k(cands, {"add_comm"}, 2) == 1.0


def test_premise_name_empty_gold_raises():
    with pytest.raises(ValueError, match="empty gold premise names"):
        premise_name_match_at_k(["simp"], set(), 1)


def test_premise_name_bad_k_raises():
    with pytest.raises(ValueError, match="k must be >= 1"):
        premise_name_match_at_k(["simp"], {"add_comm"}, 0)


def test_premise_name_is_an_upper_bound_partner_to_match():
    # Whenever the tactic matches exactly, it necessarily names the gold premise too.
    ref = "exact add_comm a b"
    assert tactic_match_at_k([ref], ref, 1) == 1.0
    assert premise_name_match_at_k([ref], {"add_comm"}, 1) == 1.0


def test_rank_and_match_at_k_agree():
    cands = ["a", "b", "c", "d"]
    rank = first_match_rank(cands, "c")
    assert rank == 3
    for k in range(1, 6):
        expected = 1.0 if rank <= k else 0.0
        assert tactic_match_at_k(cands, "c", k) == expected
