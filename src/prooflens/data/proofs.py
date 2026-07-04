"""Proof-split loader: split files -> (state_before, gold_premises) evaluation Examples.

One example = one tactic that uses >=1 (located) premise; query = `state_before`; gold = the
premises that specific tactic used (not the whole theorem). Gold is resolved exactly as ReProver's
`get_all_pos_premises`: each provenance is mapped via `corpus.locate_premise(def_path, def_pos)`
(position-containment) and provenances that cannot be located are dropped; a tactic left with no
gold is skipped entirely (ReProver `evaluate.py` does `if len(all_pos_premises) == 0: continue`).

`random` and `novel_premises` are loaded separately, never mixed (docs/EVALUATION.md §1). The
accessibility set is NOT stored here (it is large); the Example carries `file_path` and the
theorem `start` so the eval loop can compute it via `accessibility.accessible_premises`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from prooflens.data.corpus import Corpus, Pos


@dataclass(frozen=True)
class Example:
    """One evaluation example (one tactic with >=1 located gold premise)."""

    eid: str                    # f"{theorem_full_name}#{tactic_index}"
    theorem: str                # theorem full_name
    file_path: str              # theorem's file (for accessibility)
    thm_pos: Pos                # theorem start (accessibility position)
    state: str                  # state_before (the query)
    gold: frozenset[str]        # gold premise UIDs (path::full_name)


def gold_premises(corpus: Corpus, annotated_tactic) -> set[str]:
    """UIDs of the premises a tactic used, located in the corpus (unlocatable ones dropped)."""
    _, provenances = annotated_tactic
    out: set[str] = set()
    for prov in provenances:
        p = corpus.locate_premise(prov["def_path"], (prov["def_pos"][0], prov["def_pos"][1]))
        if p is not None:
            out.add(p.uid)
    return out


def load_split(
    splits_dir: str,
    split: str,
    corpus: Corpus,
    split_file: str = "test.json",
) -> Iterator[Example]:
    """Yield evaluation Examples for `split` ('random' | 'novel_premises').

    Skips tactics whose located-gold set is empty (matching ReProver's eval), and theorems with
    no tactics contribute nothing.
    """
    if split not in ("random", "novel_premises"):
        raise ValueError(f"unknown split {split!r}; expected 'random' or 'novel_premises'")
    data_path = Path(splits_dir) / split / split_file
    with open(data_path, encoding="utf-8") as fh:
        theorems = json.load(fh)

    for thm in theorems:
        thm_pos: Pos = (thm["start"][0], thm["start"][1])
        for j, tac in enumerate(thm.get("traced_tactics", [])):
            gold = gold_premises(corpus, tac["annotated_tactic"])
            if not gold:                                    # drop tactics with no located gold
                continue
            yield Example(
                eid=f'{thm["full_name"]}#{j}',
                theorem=thm["full_name"],
                file_path=thm["file_path"],
                thm_pos=thm_pos,
                state=tac["state_before"],
                gold=frozenset(gold),
            )
