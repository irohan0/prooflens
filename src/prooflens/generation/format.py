"""Generator input construction — replicated **verbatim from ReProver's `common.py`**.

The Phase-21 claim is "the generator is held fixed; only the retriever's premises change." That
claim is only meaningful if the generator receives its context in *exactly* the format the
published checkpoint (`kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small`) was trained on. A
plausible-looking but wrong concatenation would silently degrade every condition and invalidate
the comparison — the same trap Phase 7 hit with premise serialization (see
`results/LEARNINGS.md` [2026-07-03]). So this module reproduces ReProver's own
`format_augmented_state` line for line, including its quirks.

ReProver source (github.com/lean-dojo/ReProver, `common.py`)::

    def format_augmented_state(
        s: str, premises: List[Premise], max_len: Optional[int] = None, p_drop: float = 0.0
    ) -> str:
        aug_s = ""
        length = 0
        if max_len is None:
            max_len = 9999999999999999999999
        max_premises_len = max_len - len(bytes(s.encode("utf-8")))

        for p in premises:
            if random.random() < p_drop:
                continue
            p_str = f"{p.serialize()}\\n\\n"
            l = len(bytes(p_str.encode("utf-8")))
            if length + l > max_premises_len:
                continue
            length += l
            aug_s = p_str + aug_s

        aug_s += s
        return aug_s

Three behaviours here are easy to get wrong and are load-bearing:

1. **Premises are PREPENDED, so the state comes LAST.** Because `aug_s = p_str + aug_s`, the
   premises appear in *reverse* list order, which means the **highest-ranked premise ends up
   immediately before the proof state** — the position a causal reader (and the model) sees last.
   Getting the order backwards would systematically hand the model its worst candidates in the
   most salient slot.
2. **The budget is measured in UTF-8 BYTES, not tokens or characters.** `max_premises_len` is
   `max_len - len(state_bytes)`; ByT5 is a byte-level model, so bytes are the natural unit.
3. **An over-budget premise is SKIPPED, not `break`ed on.** ReProver writes `continue`, so a long
   premise is dropped and *shorter, lower-ranked* premises can still be packed in after it. A
   `break` would truncate the list at the first oversized premise and change the context.

`p_drop` is a **training-time** augmentation only: ReProver's datamodule passes
`self.p_drop if self.is_train else 0.0`. Evaluation therefore always uses `p_drop=0.0` and the
RNG is never consulted. It is implemented faithfully (with an injectable `rng` so it is
deterministic under test) rather than dropped, so this stays a true mirror of the reference.
"""

from __future__ import annotations

import random

# ReProver common.py — the self-reference markers used by `Premise.serialize()`. Re-exported from
# the retriever module so there is exactly one definition in the codebase.
from prooflens.retrievers.dense import MARK_END_SYMBOL, MARK_START_SYMBOL, serialize_premise

__all__ = [
    "MARK_END_SYMBOL",
    "MARK_START_SYMBOL",
    "format_augmented_state",
    "format_augmented_state_with_count",
    "remove_marks",
    "serialize_premise",
]

# ReProver's sentinel for "no limit" (kept literal for fidelity; only used when max_len is None).
_NO_LIMIT = 9999999999999999999999


def remove_marks(s: str) -> str:
    """Strip ReProver's `<a>` / `</a>` self-reference markers — verbatim `common.py::remove_marks`.

    Applied by ReProver's generation datamodule to build the target tactic
    (`tactic = remove_marks(tac["tactic"])`). On LeanDojo Benchmark 4 the `tactic` field is
    already mark-free (the marked variant is `annotated_tactic[0]`), so this is defensive — but
    we apply it exactly where ReProver does so the target string is byte-identical to theirs.
    """
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


def format_augmented_state(
    state: str,
    premises: list[str],
    max_len: int | None = None,
    p_drop: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    """Build the generator input from a proof state and ranked, already-serialized premises.

    Mirrors ReProver `common.py::format_augmented_state`. `premises` are premise strings **in
    retrieval rank order (best first)**, each already passed through `serialize_premise`
    (ReProver's `Premise.serialize`) by the caller — keeping this function pure and hermetically
    testable while the serialization stays shared with the dense retriever.

    Args:
        state: the proof state (`state_before`); ends up LAST in the returned string.
        premises: serialized premise texts, best-ranked first.
        max_len: total UTF-8 byte budget (ReProver's `max_inp_seq_len`, 2300 for Lean 4).
            `None` means unlimited, as in ReProver.
        p_drop: training-only premise dropout. Evaluation passes 0.0 (ReProver does the same),
            so the RNG is never consulted on the eval path.
        rng: optional `random.Random` for deterministic `p_drop` under test; defaults to the
            module-level `random`, which is what ReProver uses.

    Returns:
        The augmented state: `"<premise_n>\\n\\n...<premise_1>\\n\\n<state>"`.

    Note:
        If the state alone already exceeds `max_len`, `max_premises_len` goes negative and every
        premise is skipped — the model then sees the bare state, and the tokenizer truncates it.
        That is ReProver's behaviour and we keep it rather than "fixing" it.

    Documented deviation (output-equivalent):
        ReProver evaluates `random.random() < p_drop` unconditionally; we short-circuit when
        `p_drop == 0.0`. `random.random()` returns a value in [0, 1), so `< 0.0` is *always*
        False — the returned string is provably identical either way. The short-circuit exists
        so the eval path (which always passes `p_drop=0.0`) does not advance the global seeded
        RNG stream, keeping runs reproducible under `utils.seed.set_global_seed`. Flagged here
        because every deviation from the reference must be visible, not silent.
    """
    return format_augmented_state_with_count(state, premises, max_len, p_drop, rng)[0]


def format_augmented_state_with_count(
    state: str,
    premises: list[str],
    max_len: int | None = None,
    p_drop: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[str, int]:
    """As `format_augmented_state`, but also returns **how many premises actually fit**.

    The count matters for reporting and is easy to get wrong: passing 100 premises does not mean
    the model saw 100. Under ReProver's 2300-byte budget the median real example fits only ~25
    (measured, `results/phase_logs/phase21.md`), so the study is really about each retriever's
    top ~25. Reporting the number *offered* instead of the number *used* would overstate the
    retrieval depth being tested by 4x.

    Single implementation shared with `format_augmented_state` so the count can never drift from
    the packing it describes.
    """
    if not 0.0 <= p_drop <= 1.0:
        raise ValueError(f"p_drop must be in [0, 1], got {p_drop}")
    draw = (rng or random).random

    aug_s = ""
    length = 0
    n_used = 0
    if max_len is None:
        max_len = _NO_LIMIT
    max_premises_len = max_len - len(state.encode("utf-8"))

    for p in premises:
        if p_drop > 0.0 and draw() < p_drop:
            continue
        p_str = f"{p}\n\n"
        p_len = len(p_str.encode("utf-8"))
        if length + p_len > max_premises_len:
            continue          # ReProver uses `continue`, NOT `break` — shorter premises still fit
        length += p_len
        n_used += 1
        aug_s = p_str + aug_s  # prepend: best-ranked premise ends up adjacent to the state

    aug_s += state
    return aug_s, n_used
