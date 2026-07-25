"""Corpus IDF over ColBERT sub-word tokens — the data-derived replacement for the hand-set weight.

Phase 9's symbol weighting gives **every** symbol sub-word the same constant `w=4.0`. That is a
hand-set heuristic (README limitation 5): it cannot tell `IsSkewAdjoint` (rare, decisive) from
`add` (ubiquitous, uninformative). Phase 22 replaces the constant with each token's **inverse
document frequency** across the premise corpus — the classic measure of how discriminative a term
is — so a rare symbol is up-weighted more than a common one.

`w(i) = idf(token_i)` in the MaxSim sum, where `idf(t) = log((N+1)/(df(t)+1)) + 1` (smoothed,
always ≥ 1). `df(t)` = the number of premises whose text contains sub-word `t`; `N` = number of
premises. This module is the **pure** part — document-frequency accumulation, the IDF formula, and
the lookup table with save/load — with **no torch and no tokenizer dependency**, so it is fully
unit-testable. The tokenization that feeds it (the ColBERT sub-word tokenizer over premise text)
lives in `scripts/build_token_idf.py`, which runs on the cluster and writes the table this loads.

Design notes:
- **Keyed by token STRING** (`convert_ids_to_tokens` output), the exact form the query path
  (`_ColBERTEncoder.encode_query_with_tokens`) produces — so a query token looks up the DF of the
  identical sub-word seen in premises. Building the table with the same tokenizer is what keeps the
  two sides aligned.
- **Split-agnostic.** DF is over premise *text*, independent of any model's encoding, so one table
  serves the random and novel fine-tuned indices alike (fine-tuning does not change the vocabulary).
- **Scale-invariance.** For a fixed query the per-token weights multiply that query's MaxSim sums;
  scaling every weight by a constant leaves the ranking unchanged. So only the *relative* weights
  matter — no magic normalisation constant to tune. (This holds for the pure-`idf` mode; the
  `symbol_idf` blend mixes IDF weights with a flat default, where the IDF scale relative to that
  default does matter — hence `idf_scale` exists as an explicit, documented knob.)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

__all__ = ["TokenIDF", "document_frequencies", "idf_from_df"]


def document_frequencies(doc_token_sets: Iterable[Iterable[str]]) -> Counter[str]:
    """Count, per token, in how many documents it appears (document frequency).

    Each element of `doc_token_sets` is the tokens of one document; **duplicates within a document
    are ignored** (DF counts documents, not occurrences), so callers may pass a set or any iterable
    — this de-duplicates per document defensively.
    """
    df: Counter[str] = Counter()
    for tokens in doc_token_sets:
        df.update(set(tokens))
    return df


def idf_from_df(df: dict[str, int], n_docs: int) -> dict[str, float]:
    """Smoothed inverse document frequency: `idf(t) = log((N+1)/(df(t)+1)) + 1`.

    The `+1`s (a) never divide by zero, (b) keep every IDF ≥ 1 (a token in *every* document gets
    exactly 1, never 0 — it still contributes its raw MaxSim, just never *more* than baseline), and
    (c) give rare tokens a large but bounded weight (a df=1 token in an N≈181k corpus gets ≈ 13).
    This is the standard scikit-learn `smooth_idf` form, chosen so the numbers are recognisable.
    """
    if n_docs < 0:
        raise ValueError(f"n_docs must be non-negative, got {n_docs}")
    return {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}


class TokenIDF:
    """A loaded token→IDF table with a weight lookup, plus JSON persistence.

    `weight(token, fallback)` returns the token's IDF, or `fallback` for tokens absent from the
    table (model special tokens like `[Q]`/`[MASK]`, and any sub-word never seen in a premise — both
    of which should not receive an IDF up-weight). Kept deliberately tiny and dependency-free.
    """

    def __init__(self, idf: dict[str, float], n_docs: int, meta: dict | None = None) -> None:
        self.idf = idf
        self.n_docs = n_docs
        self.meta = meta or {}

    @classmethod
    def from_document_tokens(
        cls, doc_token_sets: Iterable[Iterable[str]], meta: dict | None = None
    ) -> TokenIDF:
        """Build directly from per-document token iterables (used by the builder and the tests)."""
        docs = [set(t) for t in doc_token_sets]
        df = document_frequencies(docs)
        idf = idf_from_df(df, len(docs))
        return cls(idf, len(docs), meta)

    def weight(self, token: str, fallback: float = 1.0) -> float:
        """IDF of `token`, or `fallback` if it is not in the table."""
        return self.idf.get(token, fallback)

    # -- persistence ---------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"n_docs": self.n_docs, "meta": self.meta, "idf": self.idf}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TokenIDF:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(blob["idf"], int(blob.get("n_docs", 0)), blob.get("meta", {}))

    def __len__(self) -> int:
        return len(self.idf)
