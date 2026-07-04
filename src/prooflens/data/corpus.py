"""Corpus loader: corpus.jsonl -> Premise objects + import graph.

Schema confirmed against the real LeanDojo Benchmark 4 (Zenodo v10) and ReProver's `common.py`
in Phase 2 (see results/phase_logs/phase2.md). Each corpus.jsonl line is one Lean file:
`{"path": str, "imports": [str], "premises": [{"full_name","code","start","end","kind"}]}`,
where `start`/`end` are `[line, col]`.

Faithful to ReProver's `Corpus`:
- premise identity is **(path, full_name, start)** (ReProver's `Premise` is `unsafe_hash` on those;
  `end`/`code` excluded from equality);
- `locate_premise(path, pos)` returns the premise whose `[start, end]` span **contains** `pos`;
- accessibility (in `accessibility.py`) uses the file import graph (transitive) + same-file
  positions. We compute transitive imports by memoised BFS rather than pulling in networkx.

Positions are Python tuples `(line, col)` so `<=` is the correct lexicographic order.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field

Pos = tuple[int, int]


@dataclass(frozen=True)
class Premise:
    """A single premise. Equality/hash = (path, full_name, start), mirroring ReProver."""

    path: str
    full_name: str
    start: Pos
    end: Pos = field(compare=False)
    code: str = field(compare=False, repr=False)
    kind: str = field(compare=False)

    @property
    def uid(self) -> str:
        """Stable, collision-free string id used by the metrics.

        `(path, full_name)` is NOT unique — the real corpus has 16 premises (macro-generated `$`
        names) that collide on it — but `(path, full_name, start)` IS unique (verified in Phase 4
        real-data validation). So `start` is encoded here, mirroring ReProver's premise identity.
        """
        return f"{self.path}::{self.full_name}@{self.start[0]},{self.start[1]}"


class Corpus:
    """The premise corpus + file import graph, loaded from corpus.jsonl."""

    def __init__(self, premises_by_path: dict[str, list[Premise]],
                 imports: dict[str, list[str]]) -> None:
        self._by_path = premises_by_path
        self._imports = imports
        self.all_premises: list[Premise] = [p for ps in premises_by_path.values() for p in ps]
        self._trans_cache: dict[str, set[str]] = {}

    # -- construction -------------------------------------------------------------------------
    @classmethod
    def from_jsonl(cls, corpus_path: str) -> Corpus:
        by_path: dict[str, list[Premise]] = {}
        imports: dict[str, list[str]] = {}
        with open(corpus_path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{corpus_path}:{line_no}: invalid JSON") from exc
                path = rec["path"]
                imports[path] = list(rec.get("imports", []))
                by_path[path] = [
                    Premise(
                        path=path,
                        full_name=p["full_name"],
                        start=(p["start"][0], p["start"][1]),
                        end=(p["end"][0], p["end"][1]),
                        code=p.get("code", ""),
                        kind=p.get("kind", ""),
                    )
                    for p in rec.get("premises", [])
                ]
        return cls(by_path, imports)

    # -- lookups ------------------------------------------------------------------------------
    def get_premises(self, path: str) -> list[Premise]:
        return self._by_path.get(path, [])

    @property
    def paths(self) -> list[str]:
        return list(self._by_path)

    def locate_premise(self, path: str, pos: Pos) -> Premise | None:
        """Premise whose span contains `pos` (ReProver `locate_premise`); None if absent."""
        for p in self.get_premises(path):
            if p.start <= pos <= p.end:
                return p
        return None

    def transitive_imports(self, path: str) -> set[str]:
        """Set of files transitively imported by `path` (memoised BFS over the import graph).

        Equivalent to `transitive_dep_graph.successors(path)` in ReProver. Excludes `path`
        itself; cycle-safe via the `seen` set (Lean imports are acyclic anyway).
        """
        cached = self._trans_cache.get(path)
        if cached is not None:
            return cached
        seen: set[str] = set()
        queue = deque(self._imports.get(path, []))
        while queue:
            f = queue.popleft()
            if f in seen:
                continue
            seen.add(f)
            queue.extend(self._imports.get(f, []))
        self._trans_cache[path] = seen
        return seen

    def __len__(self) -> int:
        return len(self.all_premises)


def load_corpus(corpus_path: str) -> Corpus:
    """Load premises and the file import graph from the benchmark corpus.jsonl."""
    return Corpus.from_jsonl(corpus_path)
