"""Accessibility filter: the set of premises accessible to a given proof state.

Confirmed against ReProver `Corpus.get_accessible_premises` (Phase 2): a premise is accessible to
a theorem at file `path` and position `pos` iff it is

  * defined earlier in the **same file** (`premise.end <= pos`), or
  * defined in a **transitively imported** file.

The accessibility position is the **theorem's start** (`Pos(*thm["start"])`), which ReProver uses
as `ctx.theorem_pos`. This SAME filter is applied identically to every retriever, or the
comparison is invalid (docs/ARCHITECTURE.md §7). Returns premise UIDs (`path::full_name`).
"""

from __future__ import annotations

from prooflens.data.corpus import Corpus, Pos


def accessible_premises(corpus: Corpus, path: str, pos: Pos) -> set[str]:
    """UIDs of premises accessible to a theorem at (`path`, `pos`)."""
    accessible: set[str] = {
        p.uid for p in corpus.get_premises(path) if p.end <= pos
    }
    for imported in corpus.transitive_imports(path):
        accessible.update(p.uid for p in corpus.get_premises(imported))
    return accessible
