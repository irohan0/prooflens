"""Pure, unit-tested metrics for generated tactics (Phase 21's scoring floor).

Retrieval metrics (`eval/metrics.py`) score a ranked list of premises. These score a ranked list
of **generated tactic strings** against the one tactic the human actually wrote, which is the
downstream signal Part 4 is after: does better premise selection make the generator write the
right tactic more often?

What "correct" means here — and its limits (stated up front, because this bounds the claim):

- We report **exact string match** between a generated tactic and the ground-truth tactic. This
  is a *lower bound* on correctness: Lean tactics that differ textually can be semantically
  equivalent (`simp [foo]` vs `simp only [foo]`, argument reordering, different-but-valid term
  proofs), and a tactic that does not match the human's can still close the goal. Only a real
  prover (Phase 20's blocked path) can measure "does it prove the theorem."
- Because it is a lower bound applied **identically to every retriever condition**, it remains a
  fair *relative* comparison — which is the whole point of the fixed-generator design. We must
  not report it as proof success.
- We report both a **strict** (byte-identical) and a **normalized** variant. Normalization only
  collapses whitespace, which Lean's surface syntax treats as insignificant between tokens. The
  strict number is kept so the effect of normalization is always visible rather than assumed.

Conventions:
- `candidates` is best-first (beam order); duplicates are deduped keeping the best rank, exactly
  as `metrics.py` does for premises, so one repeated string never occupies two slots.
- An empty `reference` is not a valid example and raises, mirroring `metrics.py`'s loud-failure
  policy for empty gold.
- `k` is 1-indexed and must be >= 1.
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_tactic(tactic: str) -> str:
    """Collapse all whitespace runs to a single space and strip the ends.

    Deliberately conservative: it does NOT reorder arguments, drop `only`, canonicalise brackets,
    or otherwise try to decide semantic equivalence. Whitespace between tokens is insignificant
    in Lean's surface syntax, so this removes formatting noise (a newline in a generated tactic
    vs a space in the reference) without ever making two genuinely different tactics compare
    equal. Anything cleverer would risk inflating the score, which we refuse to do.
    """
    return _WHITESPACE.sub(" ", tactic).strip()


def _dedupe(candidates: list[str]) -> list[str]:
    """`candidates` with duplicates removed, each kept at its first (best) rank."""
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _validate(reference: str, k: int | None = None) -> None:
    if not reference:
        raise ValueError(
            "empty reference tactic: not a valid generation example "
            "(every eval example is a real tactic — check split loading)"
        )
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def tactic_match_at_k(
    candidates: list[str],
    reference: str,
    k: int,
    normalize: bool = True,
) -> float:
    """1.0 if any of the top-`k` generated tactics equals `reference`, else 0.0.

    At `k=1` this is top-1 accuracy (the headline: did the generator's single best guess match).
    At `k>1` it is the standard "any of the beam" accuracy — a useful secondary number, since a
    real prover would try several candidates, but it must never be reported as the headline
    because a prover also has to *pick* the right one.
    """
    _validate(reference, k)
    ranked = _dedupe(candidates)[:k]
    if normalize:
        ref = normalize_tactic(reference)
        return 1.0 if any(normalize_tactic(c) == ref for c in ranked) else 0.0
    return 1.0 if any(c == reference for c in ranked) else 0.0


def premise_name_match_at_k(
    candidates: list[str],
    gold_short_names: set[str],
    k: int,
) -> float:
    """1.0 if any of the top-`k` generated tactics *names a gold premise*, else 0.0.

    **Why this metric exists** (added after the Phase-21 pilot, which made the need concrete).
    The pilot produced, for a real example::

        reference : rw [mem_skewAdjointSubmodule] at *
        generated : rw [mem_skewAdjointSubmodule] at hf hg

    Exact match scores that 0 — yet the generator picked exactly the right lemma and differed only
    in a location specifier. Since this project is about **premise selection**, "did the generator
    name the premise the human used" is the more faithful downstream signal, and it is far less
    brittle than requiring the entire tactic to match character for character.

    `gold_short_names` are last-dotted-component names (`Nat.add_comm` -> `add_comm`), matched as
    whole Lean tokens using the **same tokenizer and short-name rule** as the Phase-11 audit and
    the Phase-19 lexical stratification, so the numbers stay comparable across phases. Substring
    matching is deliberately avoided: `add_comm` must not be credited by `add_comm_sub`.

    Read it as an UPPER bound partner to `match@k`'s lower bound: naming the right lemma is
    necessary but not sufficient for a correct tactic. Neither is a proof-success rate.
    """
    if not gold_short_names:
        raise ValueError(
            "empty gold premise names: not a valid example (every example has >=1 gold premise)"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Lazy import: bm25 pulls in rank_bm25 at module level, and this module must stay importable
    # (and hermetically testable) without it.
    from prooflens.retrievers.bm25 import lean_tokenize

    for cand in _dedupe(candidates)[:k]:
        if gold_short_names & set(lean_tokenize(cand)):
            return 1.0
    return 0.0


def first_match_rank(
    candidates: list[str],
    reference: str,
    normalize: bool = True,
) -> int | None:
    """1-indexed rank of the first generated tactic equal to `reference`; None if none match.

    Persisted per example so any `match@k` can be recomputed post-hoc without re-running the
    generator (the same reason `evaluate.py` persists the full top-100 premise list).
    """
    _validate(reference)
    ref = normalize_tactic(reference) if normalize else reference
    for rank, c in enumerate(_dedupe(candidates), start=1):
        if (normalize_tactic(c) if normalize else c) == ref:
            return rank
    return None
