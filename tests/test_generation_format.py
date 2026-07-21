"""Hermetic tests for the ReProver-verbatim generator input format.

These pin the three behaviours that a plausible-but-wrong reimplementation would get wrong
(premise ORDER, BYTE budget, skip-not-break) plus the mark stripping. If any of these break, the
Phase-21 comparison is invalid even though every number would still "look reasonable" — which is
exactly why they are asserted explicitly rather than left to a smoke test.
"""

from __future__ import annotations

import random

import pytest

from prooflens.generation.format import (
    format_augmented_state,
    format_augmented_state_with_count,
    remove_marks,
    serialize_premise,
)

# -- remove_marks ---------------------------------------------------------------------------------

def test_remove_marks_strips_both_markers():
    assert remove_marks("dsimp [<a>invFunIdAssoc</a>]") == "dsimp [invFunIdAssoc]"


def test_remove_marks_is_a_noop_on_plain_tactics():
    # Benchmark 4's `tactic` field is already mark-free; ReProver still calls remove_marks on it.
    assert remove_marks("aesop_cat") == "aesop_cat"


# -- ordering: premises are PREPENDED, so best-ranked sits next to the state -----------------------

def test_state_comes_last_and_best_premise_is_adjacent_to_it():
    out = format_augmented_state("STATE", ["P1", "P2", "P3"])
    # aug_s = "P3\n\n" + "P2\n\n" + "P1\n\n", then += state
    assert out == "P3\n\nP2\n\nP1\n\nSTATE"
    assert out.endswith("STATE")
    # the highest-ranked premise (P1, first in the list) is the one immediately before the state
    assert out.endswith("P1\n\nSTATE")


def test_single_premise_layout():
    assert format_augmented_state("S", ["ONLY"]) == "ONLY\n\nS"


def test_no_premises_returns_the_bare_state():
    assert format_augmented_state("S", []) == "S"


# -- budget is in UTF-8 BYTES, not characters -----------------------------------------------------

def test_budget_counts_utf8_bytes_not_characters():
    # A Lean state full of unicode: '⊢' is 3 bytes but 1 character. If the budget were measured
    # in characters the premise would fit; in bytes it must not.
    state = "⊢" * 10                       # 10 chars, 30 bytes
    premise = "x" * 5                       # premise string -> "xxxxx\n\n" = 7 bytes
    assert len(state) == 10 and len(state.encode("utf-8")) == 30
    # max_len 33 -> byte budget = 33 - 30 = 3, which is < 7, so the premise is dropped.
    assert format_augmented_state(state, [premise], max_len=33) == state
    # max_len 37 -> budget 7, exactly enough.
    assert format_augmented_state(state, [premise], max_len=37) == f"{premise}\n\n{state}"


def test_budget_boundary_is_strict_greater_than():
    # ReProver skips when `length + l > max_premises_len`; equality must still fit.
    state = "S"                                    # 1 byte
    premise = "ab"                                 # "ab\n\n" = 4 bytes
    assert format_augmented_state(state, [premise], max_len=5) == "ab\n\nS"   # budget exactly 4
    assert format_augmented_state(state, [premise], max_len=4) == "S"        # budget 3 -> dropped


# -- over-budget premise is SKIPPED, not break ----------------------------------------------------

def test_oversized_premise_is_skipped_and_later_shorter_ones_still_fit():
    # This is the `continue` vs `break` distinction. Budget fits the short premise but not the
    # long one; a `break` implementation would return just the state.
    state = "S"                       # 1 byte
    long_p = "L" * 50                 # 52 bytes with separators
    short_p = "S1"                    # 4 bytes with separators
    out = format_augmented_state(state, [long_p, short_p], max_len=11)  # budget = 10
    assert out == "S1\n\nS", "long premise must be skipped while the shorter later one is kept"
    assert long_p not in out


def test_state_larger_than_budget_drops_all_premises():
    # Negative budget -> every premise skipped; the bare state is returned (ReProver behaviour).
    out = format_augmented_state("x" * 100, ["P"], max_len=10)
    assert out == "x" * 100


# -- max_len=None means unlimited -----------------------------------------------------------------

def test_none_max_len_keeps_every_premise():
    premises = [f"P{i}" for i in range(50)]
    out = format_augmented_state("S", premises, max_len=None)
    for p in premises:
        assert f"{p}\n\n" in out


# -- p_drop (training-only) -----------------------------------------------------------------------

def test_p_drop_zero_keeps_everything_and_never_consumes_rng():
    class ExplodingRandom(random.Random):
        def random(self) -> float:            # pragma: no cover - must never be called
            raise AssertionError("p_drop=0.0 must not consume the RNG")

    out = format_augmented_state("S", ["A", "B"], p_drop=0.0, rng=ExplodingRandom())
    assert out == "B\n\nA\n\nS"


def test_p_drop_one_drops_every_premise():
    out = format_augmented_state("S", ["A", "B"], p_drop=1.0, rng=random.Random(0))
    assert out == "S"


def test_p_drop_is_deterministic_under_a_seeded_rng():
    a = format_augmented_state("S", [f"P{i}" for i in range(20)], p_drop=0.5, rng=random.Random(7))
    b = format_augmented_state("S", [f"P{i}" for i in range(20)], p_drop=0.5, rng=random.Random(7))
    assert a == b


def test_invalid_p_drop_raises():
    with pytest.raises(ValueError, match="p_drop"):
        format_augmented_state("S", ["A"], p_drop=1.5)


# -- serialization is shared with the dense retriever ---------------------------------------------

def test_serialize_premise_marks_the_self_reference():
    # Same function the dense retriever uses -> the generator sees ReProver's exact premise text.
    out = serialize_premise("Nat.add_comm", "theorem Nat.add_comm (a b : Nat) : a + b = b + a")
    assert "<a>Nat.add_comm</a>" in out


# -- the fitted-premise count (what the model ACTUALLY saw) ---------------------------------------

def test_count_reports_premises_that_fit_not_premises_offered():
    # The bug this guards: reporting len(premises) would claim 3 when only 1 fit.
    state = "S"                       # 1 byte
    out, n = format_augmented_state_with_count(
        state, ["ab", "L" * 50, "M" * 50], max_len=5      # budget 4 -> only "ab\n\n" fits
    )
    assert out == "ab\n\nS"
    assert n == 1, "must count what survived the byte budget, not what was offered"


def test_count_is_zero_when_nothing_fits():
    _, n = format_augmented_state_with_count("x" * 100, ["P", "Q"], max_len=10)
    assert n == 0


def test_count_matches_all_when_unlimited():
    premises = [f"P{i}" for i in range(7)]
    out, n = format_augmented_state_with_count("S", premises, max_len=None)
    assert n == 7
    assert out.endswith("S")


def test_count_and_string_come_from_the_same_packing():
    # The two must never drift: the count equals the number of separators added.
    premises = [f"P{i}" for i in range(40)]
    out, n = format_augmented_state_with_count("STATE", premises, max_len=120)
    assert n == out.count("\n\n")
    assert format_augmented_state("STATE", premises, max_len=120) == out


def test_end_to_end_shape_matches_reprover_layout():
    """One realistic assembly: serialized premises, best-first, state last."""
    p1 = serialize_premise("add_comm", "theorem add_comm : a + b = b + a")
    p2 = serialize_premise("mul_comm", "theorem mul_comm : a * b = b * a")
    state = "a b : Nat\n⊢ a + b = b + a"
    out = format_augmented_state(state, [p1, p2], max_len=2300)
    assert out == f"{p2}\n\n{p1}\n\n{state}"
    assert out.endswith(state)
