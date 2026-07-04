"""BM25 sparse lexical baseline (no training).

The field's weak baseline. ReProver ships no BM25, so this is our own implementation; the published
BM25 numbers come from the LeanDojo paper (arXiv 2306.15626) and our BM25 only needs to land in
their rough range (diagnostic #1 in results/comparison.md). Every choice that could affect
comparability is a documented flag, surfaced in the results provenance header.

Design (docs/ARCHITECTURE.md §4):
- **Lean-aware tokenizer** (`lean_tokenize`): keep identifiers/operators/type constructors as
  tokens, split dotted names into components, do not lowercase declaration-name casing. Also reused
  by the late-interaction symbol weighting later (ARCHITECTURE.md §6).
- **Premise document** = `full_name + " " + code` (name gives the namespaced identifier tokens,
  code gives the statement). The dense retriever uses ReProver's marker serialization instead; the
  two methods legitimately use their own natural text representation.
- **Query** = the proof state (`state_before`), matching ReProver's `Context.serialize` (state).
- Score the **accessible** subset at query time (identical filter for every retriever).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from prooflens.retrievers.base import Retriever

# Identifier runs (ASCII letters/digits/underscore/prime) OR any single other non-space char
# (so unicode math operators and `.` become individual tokens; dotted names split on `.`).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+|[^\sA-Za-z0-9_']", re.UNICODE)


def lean_tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Lean-aware tokenization. `lowercase=False` by default (declaration-name casing matters)."""
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def premise_document(full_name: str, code: str) -> str:
    """The text a premise is indexed as for BM25: fully-qualified name then its source."""
    return f"{full_name} {code}"


class BM25Retriever(Retriever):
    def __init__(self, lowercase: bool = False, index_dir: str | None = None) -> None:
        self.lowercase = lowercase
        self.index_dir = index_dir
        self._bm25: BM25Okapi | None = None
        self._uids: list[str] = []
        self._uid_to_idx: dict[str, int] = {}

    # -- index ---------------------------------------------------------------------------------
    def _index_path(self) -> Path | None:
        return Path(self.index_dir) / "bm25.pkl" if self.index_dir else None

    def build_index(self, corpus, rebuild: bool = False) -> None:
        """Tokenise every premise document and build the BM25 index. Reload a persisted index if
        present (unless `rebuild`)."""
        path = self._index_path()
        if path is not None and path.exists() and not rebuild:
            self._load(path)
            return

        self._uids = [p.uid for p in corpus.all_premises]
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}
        tokenised_docs = [
            lean_tokenize(premise_document(p.full_name, p.code), self.lowercase)
            for p in corpus.all_premises
        ]
        # rank_bm25 requires every document to have >=1 token; guard the rare empty doc.
        tokenised_docs = [doc if doc else ["<empty>"] for doc in tokenised_docs]
        self._bm25 = BM25Okapi(tokenised_docs)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({"bm25": self._bm25, "uids": self._uids}, fh)

    def _load(self, path: Path) -> None:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self._bm25 = blob["bm25"]
        self._uids = blob["uids"]
        self._uid_to_idx = {uid: i for i, uid in enumerate(self._uids)}

    # -- retrieve ------------------------------------------------------------------------------
    def retrieve(self, state: str, accessible: set[str], k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("build_index must be called before retrieve")
        query = lean_tokenize(state, self.lowercase)
        # Score ONLY the accessible premises (the candidate set). BM25 scores are per-document
        # independent — global IDF, per-doc TF — so scoring the accessible subset via
        # get_batch_scores is identical to scoring all premises then filtering, but far faster
        # (the corpus has ~180k premises; the accessible set is a fraction of that).
        acc_uids = [uid for uid in accessible if uid in self._uid_to_idx]
        acc_idx = [self._uid_to_idx[uid] for uid in acc_uids]
        scores = self._bm25.get_batch_scores(query, acc_idx)
        candidates = list(zip(acc_uids, (float(s) for s in scores), strict=True))
        # Sort by score desc; tie-break by uid for determinism.
        candidates.sort(key=lambda t: (-t[1], t[0]))
        return candidates[:k]
