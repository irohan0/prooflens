"""Retriever interface. Every retriever is swappable behind this ABC; the eval harness never
knows which one it is scoring (docs/ARCHITECTURE.md §3).

STUB — interface sketched; concrete signatures to be finalised in the planning phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def build_index(self, corpus, rebuild: bool = False) -> None:
        """Precompute over the full premise corpus and persist to the configured index dir.
        Reload if present unless `rebuild` is True."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, state: str, accessible: set[str], k: int) -> list[tuple[str, float]]:
        """Return up to k (premise_full_name, score) ranked best-first, restricted to
        `accessible`. The accessibility restriction is applied identically for every
        retriever, or the comparison is invalid."""
        raise NotImplementedError
