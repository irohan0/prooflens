"""Model-free data-audit statistics for the pre-fine-tuning phase (Phase 11).

Pure functions over the existing loaders (corpus, proofs, accessibility) and the Lean-aware BM25
tokenizer that measure the properties of the benchmark which govern how well fine-tuning can work:

- **Retrieval ceiling** — is every gold premise present and accessible? (caps achievable recall).
- **Pair volume & sizes** — how many (state, gold) training positives, and how big are the gold and
  accessible sets (informs batch size, in-batch vs hard negatives).
- **Premise-frequency distribution** — do a few premises dominate the positives? (dedup / capping).
- **Lexical signal** — how often does the gold premise's name appear literally in the state? (the
  signal BM25 hard-negatives exploit; measured on test in Phase 6 as 26.6% / 19.4%).

No torch / transformers / pylate here, so this runs locally on the real corpus and is hermetically
unit-tested. The tokenizer- and encoder-side audit (unicode fragmentation, truncation, symbol-token
fraction) and the symbol-weight / query-length sweeps need the model and live in `scripts/audit.py`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import Corpus
from prooflens.data.proofs import Example, examples_from_theorems
from prooflens.retrievers.bm25 import lean_tokenize


def uid_full_name(uid: str) -> str:
    """Recover the premise `full_name` from a UID `"{path}::{full_name}@{line},{col}"`.

    File paths carry no `::` and Lean names carry no `@`, so splitting on the first `::` and the
    last `@` is unambiguous (premise UID format is defined in `corpus.Premise.uid`).
    """
    return uid.split("::", 1)[1].rsplit("@", 1)[0]


def short_name(full_name: str) -> str:
    """The last dotted component of a fully-qualified name (`Nat.add_comm` -> `add_comm`)."""
    return full_name.rsplit(".", 1)[-1]


def _stats(values: list[int]) -> dict:
    """min / mean / median / p90 / max / n for a list of sizes (None-safe on empty)."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "p90": None, "max": None}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": sum(s) / n,
        "median": s[n // 2],
        "min": s[0],
        "p90": s[min(n - 1, int(0.9 * n))],
        "max": s[-1],
    }


def _stream_json_array(path: str, max_items: int | None = None,
                       chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Stream objects from a top-level JSON array without materialising the whole file.

    Uses `json.raw_decode` (NOT brace counting) so it is robust to braces/quotes inside strings —
    Lean `code` fields contain literal `{ }` and quotes. Lets the audit run over a 350 MB
    `train.json` at bounded memory (one object + one chunk in the buffer at a time).
    """
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as fh:
        buf = ""
        while "[" not in buf:                       # advance to the opening bracket
            chunk = fh.read(chunk_size)
            if not chunk:
                return
            buf += chunk
        buf = buf[buf.index("[") + 1:]
        n = 0
        while True:
            buf = buf.lstrip()
            while buf[:1] == ",":                    # commas between array items
                buf = buf[1:].lstrip()
            if buf[:1] == "]":                       # end of the array
                return
            if not buf:
                chunk = fh.read(chunk_size)
                if not chunk:
                    return
                buf += chunk
                continue
            try:
                obj, idx = dec.raw_decode(buf)
            except json.JSONDecodeError:             # object not fully in the buffer yet
                chunk = fh.read(chunk_size)
                if not chunk:
                    return                           # truncated / malformed tail: stop cleanly
                buf += chunk
                continue
            yield obj
            n += 1
            if max_items is not None and n >= max_items:
                return
            buf = buf[idx:]


def stream_examples(path: str, corpus: Corpus,
                    max_theorems: int | None = None) -> Iterator[Example]:
    """Yield `proofs.Example` from a split JSON file, streaming theorem objects (see
    `_stream_json_array`). Reuses `proofs.examples_from_theorems` so the per-tactic rule is
    identical to `load_split`. `max_theorems` bounds the pass for a fast local sample."""
    return examples_from_theorems(_stream_json_array(path, max_theorems), corpus)


def frequency_summary(freq: Counter, top: int = 25) -> dict:
    """Summarise the premise-as-positive frequency distribution (dedup / capping decision)."""
    total = sum(freq.values())
    counts = sorted(freq.values(), reverse=True)
    n_unique = len(counts)

    def head_coverage(frac: float) -> float | None:
        if not total:
            return None
        k = max(1, int(n_unique * frac))
        return sum(counts[:k]) / total

    return {
        "n_unique_gold_premises": n_unique,
        "n_positive_pairs": total,
        "max_count": counts[0] if counts else 0,
        "median_count": counts[n_unique // 2] if counts else 0,
        "singleton_fraction": (counts.count(1) / n_unique) if n_unique else None,
        "head_coverage_top1pct": head_coverage(0.01),
        "head_coverage_top5pct": head_coverage(0.05),
        "top_premises": [(short_name(uid_full_name(u)), c) for u, c in freq.most_common(top)],
    }


def audit_examples(examples: Iterable[Example], corpus: Corpus | None = None) -> dict:
    """Single pass over Examples → data-audit dict.

    Computes pair volume, gold-size stats, the lexical (gold-name-in-state) rate, and the premise
    frequency distribution. If `corpus` is given, also computes accessibility stats (accessible-set
    sizes and the gold⊆accessible rate) — the retrieval ceiling — caching the accessible set per
    theorem (Examples arrive theorem-consecutively), matching the eval loop's bounded caching.
    """
    gold_sizes: list[int] = []
    acc_sizes: list[int] = []
    freq: Counter = Counter()
    n_examples = 0
    n_gold_in_state = 0
    n_gold_in_acc = 0
    n_acc_checked = 0
    last_key: tuple[str, tuple[int, int]] | None = None
    last_acc: set[str] = set()

    for ex in examples:
        n_examples += 1
        gold_sizes.append(len(ex.gold))
        freq.update(ex.gold)
        toks = set(lean_tokenize(ex.state))
        if any(short_name(uid_full_name(u)) in toks for u in ex.gold):
            n_gold_in_state += 1
        if corpus is not None:
            key = (ex.file_path, ex.thm_pos)
            if key != last_key:
                last_acc = accessible_premises(corpus, ex.file_path, ex.thm_pos)
                last_key = key
            acc_sizes.append(len(last_acc))
            n_acc_checked += 1
            if ex.gold <= last_acc:
                n_gold_in_acc += 1

    return {
        "n_examples": n_examples,
        "n_positive_pairs": sum(gold_sizes),
        "gold_size": _stats(gold_sizes),
        "gold_name_in_state_rate": (n_gold_in_state / n_examples) if n_examples else None,
        "accessible_size": _stats(acc_sizes),
        "gold_in_accessible_rate": (n_gold_in_acc / n_acc_checked) if n_acc_checked else None,
        "premise_frequency": frequency_summary(freq),
    }
