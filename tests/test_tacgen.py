"""Hermetic tests for the tactic generator's post-processing.

`ByT5TacticGenerator` itself needs torch + the checkpoint (cluster-only, covered by the smoke in
`scripts/tacgen_smoke.py`). The part that decides *what gets compared to the reference* —
mark-stripping and beam deduplication — is pure, so it is pinned here instead of being trusted.
"""

from __future__ import annotations

import pytest

from prooflens.generation.tacgen import TacticGenerator, dedupe_candidates


def test_marks_are_stripped_from_generated_text():
    # ReProver applies remove_marks to the DECODED output; the reference is mark-free, so
    # skipping this would silently fail otherwise-correct matches.
    out = dedupe_candidates(["exact <a>add_comm</a> a b"], [-0.1])
    assert out == [("exact add_comm a b", -0.1)]


def test_duplicates_are_dropped_keeping_the_best_score():
    # Beam output is score-ordered, so the first occurrence carries the best score.
    out = dedupe_candidates(["simp", "simp", "ring"], [-0.1, -0.5, -0.9])
    assert out == [("simp", -0.1), ("ring", -0.9)]


def test_marks_are_stripped_before_deduplication():
    # "<a>x</a>" and "x" collapse to the same tactic and must count once.
    out = dedupe_candidates(["exact <a>foo</a>", "exact foo"], [-0.2, -0.7])
    assert out == [("exact foo", -0.2)]


def test_order_is_preserved():
    out = dedupe_candidates(["c", "a", "b"], [-0.1, -0.2, -0.3])
    assert [t for t, _ in out] == ["c", "a", "b"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="scores"):
        dedupe_candidates(["a", "b"], [-0.1])


def test_empty_input_is_empty_output():
    assert dedupe_candidates([], []) == []


def test_protocol_is_satisfied_by_a_minimal_fake():
    class Fake:
        def generate(self, state: str, num_samples: int) -> list[tuple[str, float]]:
            return [("simp", 0.0)]

    assert isinstance(Fake(), TacticGenerator)
