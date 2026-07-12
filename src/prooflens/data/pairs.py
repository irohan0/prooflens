"""Training-pair construction for fine-tuning the retrievers (Phase 12).

Turns a split's `train.json` (or `val.json`) into `(query, positive, negatives[])` triplets:

- **query**  = the proof state (`state_before`), exactly the eval query.
- **positive** = a gold premise the tactic used, serialised with `premise_document` (`full_name +
  code`) — the SAME text the LI/BM25 index uses, so train text == index text == eval text.
- **negatives** = accessible, non-gold premises mined per the configured strategy.

**Negative strategies (configurable — the user-approved "both"):**
- `"bm25"`  — hard negatives: the top BM25 hits for the state (via the fast `bm25s`, corpus-wide),
  filtered to *accessible & non-gold*, then a **rank band** (skip the very top few, which are often
  unlabelled true positives → false negatives) — the de-noising policy from the Phase-11 audit.
- `"random"` — accessible non-gold sampled uniformly (the ablation baseline; needs no `bm25s`).

**Integrity (because we caught the cross-split leakage):**
- The builder reads `train.json` / `val.json` only and **refuses `test.json`** (unit-tested guard).
- Negatives exclude every premise that is gold for **any** tactic of the same theorem (not just the
  current state), so a lemma used elsewhere in the proof is never a "negative".
- **Head-capping** (Phase-11 finding: top-1% of premises = 33% of positives) keeps only the first
  `cap` `(state, positive)` pairs of each over-frequent positive so training isn't dominated by
  `rfl`/`mul_comm` — deterministic and never zeroes a premise; **dedup** collapses identical
  `(state, positive)` pairs. (Only negative mining is seeded/random.)

`bm25s` is a *mining-only* fast BM25 (≈26 ms/query vs ≈27 s with `rank_bm25` over the ~76k-premise
accessible set). It is imported lazily (like torch/pylate elsewhere) so this module and the hermetic
tests do not require it; the eval BM25 keeps `rank_bm25` (calibrated), untouched.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from prooflens.data.audit import stream_theorems
from prooflens.data.corpus import Corpus, Pos
from prooflens.data.proofs import examples_from_theorems
from prooflens.retrievers.bm25 import lean_tokenize, premise_document
from prooflens.utils.logging import get_logger

log = get_logger("pairs")


# -- accessibility as an O(1) membership test (no 76k-set materialised per theorem) ---------------

def make_accessible_check(corpus: Corpus, path: str, pos: Pos,
                          uid_to_premise: dict) -> Callable[[str], bool]:
    """Return `is_accessible(uid)` for a theorem at (`path`, `pos`), matching `accessibility.py`:
    a premise is accessible iff defined in a (transitively) imported file, or earlier in the same
    file (`end <= pos`). Uses the memoised transitive-import *set* for O(1) checks, so we never
    build the full ~76k accessible set — we only test the handful of mined candidates."""
    imports = corpus.transitive_imports(path)          # memoised set of files
    def is_accessible(uid: str) -> bool:
        p = uid_to_premise.get(uid)
        if p is None:
            return False
        return p.path in imports or (p.path == path and p.end <= pos)
    return is_accessible


# -- negative miners ------------------------------------------------------------------------------

def select_negatives_from_ranking(ranked: list[str], is_accessible: Callable[[str], bool],
                                   exclude: set[str], n_neg: int, rng: random.Random,
                                   skip: int = 2, window: int = 50) -> list[str]:
    """Pure band-selection over a BM25-ranked uid list (unit-tested without `bm25s`).

    Keep the accessible, non-excluded candidates in rank order; **skip the top `skip`** (likely
    unlabelled true positives) and sample `n_neg` from the next `window` (the hard-but-not-false
    zone). Falls back to the full candidate list if the band is too small.
    """
    cands = [u for u in ranked if u not in exclude and is_accessible(u)]
    band = cands[skip:skip + window]
    pool = band if len(band) >= n_neg else cands
    if len(pool) <= n_neg:
        return list(pool)
    return rng.sample(pool, n_neg)


class RandomMiner:
    """Uniform accessible non-gold negatives (the ablation strategy). Rejection-samples corpus uids
    (accessible ≈ 40% of the corpus, so ~2-3 tries per pick) — needs no BM25 index."""

    strategy = "random"

    def __init__(self, corpus: Corpus) -> None:
        self._uids = [p.uid for p in corpus.all_premises]

    def mine(self, query: str, is_accessible: Callable[[str], bool], exclude: set[str],
             n_neg: int, rng: random.Random, **_: object) -> list[str]:
        picks: list[str] = []
        seen: set[str] = set()
        cap_attempts = max(n_neg * 50, 50)
        for _ in range(cap_attempts):
            if len(picks) >= n_neg:
                break
            u = self._uids[rng.randrange(len(self._uids))]
            if u in exclude or u in seen or not is_accessible(u):
                continue
            seen.add(u)
            picks.append(u)
        return picks


class BM25Miner:
    """Hard negatives via `bm25s` (fast, corpus-wide), band-selected + accessibility-filtered.

    Builds ONE BM25 index over the whole corpus (tokenised with the SAME `lean_tokenize` as the eval
    BM25). Per state it retrieves the top `top_n` corpus hits once (cached across a state's several
    positives), then `select_negatives_from_ranking` filters to accessible & non-gold and applies
    the rank band. A `RandomMiner` tops up if a query yields too few hard negatives.
    """

    strategy = "bm25"

    def __init__(self, corpus: Corpus, top_n: int = 200, skip: int = 2, window: int = 50) -> None:
        import bm25s  # lazy: mining-only dependency

        self._uids = [p.uid for p in corpus.all_premises]
        tokens = [lean_tokenize(premise_document(p.full_name, p.code)) for p in corpus.all_premises]
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens)
        self.top_n = top_n
        self.skip = skip
        self.window = window
        self._fallback = RandomMiner(corpus)
        self._cache_q: str | None = None
        self._cache_ranked: list[str] = []

    def _rank(self, query: str) -> list[str]:
        if query == self._cache_q:
            return self._cache_ranked
        qtok = lean_tokenize(query)
        if not qtok:
            ranked: list[str] = []
        else:
            k = min(self.top_n, len(self._uids))
            results, _ = self._bm25.retrieve([qtok], k=k, show_progress=False)
            ranked = [self._uids[int(i)] for i in results[0]]
        self._cache_q, self._cache_ranked = query, ranked
        return ranked

    def mine(self, query: str, is_accessible: Callable[[str], bool], exclude: set[str],
             n_neg: int, rng: random.Random, **_: object) -> list[str]:
        ranked = self._rank(query)
        picks = select_negatives_from_ranking(
            ranked, is_accessible, exclude, n_neg, rng, self.skip, self.window
        )
        if len(picks) < n_neg:                     # top up with random accessible negatives
            extra_exclude = exclude | set(picks)
            picks += self._fallback.mine(
                query, is_accessible, extra_exclude, n_neg - len(picks), rng
            )
        return picks


def build_miner(negatives: str, corpus: Corpus, *, top_n: int = 200,
                skip: int = 2, window: int = 50):
    if negatives == "random":
        return RandomMiner(corpus)
    if negatives == "bm25":
        return BM25Miner(corpus, top_n=top_n, skip=skip, window=window)
    raise ValueError(f"unknown negatives strategy {negatives!r} (expected 'bm25' or 'random')")


# -- triplet construction -------------------------------------------------------------------------

def _pair_key(state: str, positive_uid: str) -> bytes:
    """Stable (cross-run) dedup key for a `(state, positive)` pair — avoids holding raw states."""
    h = hashlib.blake2b(digest_size=16)
    h.update(state.encode("utf-8"))
    h.update(b"\x00")
    h.update(positive_uid.encode("utf-8"))
    return h.digest()


def build_triplets(corpus: Corpus, split_path: str, *, negatives: str = "bm25", n_neg: int = 3,
                   cap: int | None = None, seed: int = 42, top_n: int = 200, neg_skip: int = 2,
                   neg_window: int = 50, miner: object | None = None,
                   max_theorems: int | None = None) -> Iterator[dict]:
    """Yield `{"query", "positive", "negatives": [...]}` triplets (texts) from a split file.

    Streams theorems (bounded memory), sharing one accessible-check and the theorem-gold union per
    theorem. `cap` down-samples over-frequent positive premises (seeded); identical `(state,
    positive)` pairs are de-duplicated. Refuses `test.json` (leakage guard).
    """
    name = Path(split_path).name
    if name == "test.json":
        raise ValueError(
            f"refusing to build training pairs from {name!r}: test data must never enter training"
        )

    uid_to_premise = {p.uid: p for p in corpus.all_premises}
    text_cache: dict[str, str] = {}

    def text(uid: str) -> str:
        t = text_cache.get(uid)
        if t is None:
            p = uid_to_premise[uid]
            t = premise_document(p.full_name, p.code)
            text_cache[uid] = t
        return t

    if miner is None:
        miner = build_miner(negatives, corpus, top_n=top_n, skip=neg_skip, window=neg_window)
    rng = random.Random(seed)
    seen: set[bytes] = set()
    kept: Counter = Counter()             # per positive premise: #unique pairs kept (for capping)

    for theorem in stream_theorems(split_path, max_theorems):
        path = theorem["file_path"]
        pos: Pos = (theorem["start"][0], theorem["start"][1])
        exs = list(examples_from_theorems([theorem], corpus))
        if not exs:
            continue
        theorem_gold: set[str] = set().union(*(e.gold for e in exs))
        is_accessible = make_accessible_check(corpus, path, pos, uid_to_premise)

        for ex in exs:
            for g in ex.gold:
                key = _pair_key(ex.state, g)
                if key in seen:
                    continue                              # dedup identical (state, positive)
                if cap is not None and kept[g] >= cap:
                    continue                              # head-capping: keep only the first `cap`
                seen.add(key)
                kept[g] += 1
                neg_uids = miner.mine(ex.state, is_accessible, theorem_gold, n_neg, rng)
                yield {
                    "query": ex.state,
                    "positive": text(g),
                    "negatives": [text(u) for u in neg_uids],
                }


def load_triplet_records(path: str, max_samples: int | None = None) -> Iterator[dict]:
    """Yield exploded `{"query","positive","negative"}` rows from a pairs JSONL (built above).

    Each source row `{"query","positive","negatives":[...]}` yields one triplet per negative — the
    universally-compatible contrastive format (in-batch negatives add the rest at train time). Rows
    with no negative are skipped; `max_samples` bounds the emitted triplets (the `--limit` trial).
    Shared by `train_li.py` and `train_sv.py` so both consume identical triplets — the like-for-like
    training input for the matched control.
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query, positive = row["query"], row["positive"]
            for neg in row.get("negatives", []):
                yield {"query": query, "positive": positive, "negative": neg}
                n += 1
                if max_samples is not None and n >= max_samples:
                    return
