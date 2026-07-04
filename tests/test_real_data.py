"""Real-data validation of the production loaders against LeanDojo Benchmark 4 ground truth.

Runs only when the (gitignored, 68 MB) dataset is present at
`<repo>/leandojo_data/leandojo_benchmark_4/`; skips cleanly otherwise (e.g. CI). Asserts the
PRODUCTION loaders reproduce the numbers independently established in Phase 2
(results/phase_logs/phase2.md): corpus size, collision-free UIDs, usable-example counts, mean gold
size, and the gold-is-accessible invariant.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split

BENCH = Path(__file__).parents[1] / "leandojo_data" / "leandojo_benchmark_4"

pytestmark = pytest.mark.skipif(
    not (BENCH / "corpus.jsonl").exists(),
    reason="real LeanDojo Benchmark 4 not staged locally (leandojo_data/)",
)


@functools.lru_cache(maxsize=1)
def _corpus():
    return load_corpus(str(BENCH / "corpus.jsonl"))


def test_corpus_totals_and_unique_uids():
    c = _corpus()
    assert len(c.paths) == 5674
    assert len(c) == 180973
    # UID must be collision-free (this is why start is encoded in the UID)
    uids = [p.uid for p in c.all_premises]
    assert len(set(uids)) == len(uids) == 180973


@pytest.mark.parametrize("split,expected", [("random", 2811), ("novel_premises", 4357)])
def test_usable_example_counts(split, expected):
    c = _corpus()
    examples = list(load_split(str(BENCH), split, c))
    assert len(examples) == expected


@pytest.mark.parametrize("split,mean_lo,mean_hi", [
    ("random", 2.15, 2.25),
    ("novel_premises", 2.52, 2.62),
])
def test_mean_gold_size(split, mean_lo, mean_hi):
    c = _corpus()
    examples = list(load_split(str(BENCH), split, c))
    mean_gold = sum(len(e.gold) for e in examples) / len(examples)
    assert mean_lo <= mean_gold <= mean_hi


@pytest.mark.parametrize("split", ["random", "novel_premises"])
def test_gold_subset_of_accessible(split):
    # Phase 2 invariant: every located gold premise is accessible to its theorem.
    c = _corpus()
    examples = list(load_split(str(BENCH), split, c))
    acc_cache: dict[tuple[str, tuple[int, int]], set[str]] = {}
    checked = 0
    for e in examples:
        key = (e.file_path, e.thm_pos)
        acc = acc_cache.get(key)
        if acc is None:
            acc = accessible_premises(c, e.file_path, e.thm_pos)
            acc_cache[key] = acc
        assert set(e.gold) <= acc, f"gold not accessible in {e.eid}"
        checked += 1
    assert checked == len(examples)
