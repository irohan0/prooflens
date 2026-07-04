"""End-to-end evaluation test: run the BM25 pipeline over the fixtures and check the outputs.

Asserts robust, hand-verifiable properties (not BM25's exact tie-ordering of zero-signal premises):
the unique high-IDF name token puts each gold premise at rank 1, so R@1 = (1+1+0.5)/3 and
R@10 = MRR = 1.0; plus the JSON provenance/record schema, summary.csv, and accessibility.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import load_corpus
from prooflens.eval.evaluate import evaluate

FIX = Path(__file__).parent / "fixtures" / "mini_benchmark"


def _config() -> dict:
    return {
        "name": "bm25_fixture",
        "retriever": "bm25",
        "data": {
            "corpus_path": str(FIX / "corpus.jsonl"),
            "splits_dir": str(FIX),
            "split_file": "test.json",
            "accessibility": "computed",
        },
        "tokenizer": {"lean_aware": True, "lowercase": False},
        "eval": {"k_list": [1, 10], "retrieve_k": 100, "splits": ["random"]},
        "seed": 42,
    }


def test_evaluate_end_to_end(tmp_path):
    out = evaluate(_config(), results_dir=str(tmp_path))
    agg = out["random"]

    # aggregate metrics (robust: each gold's unique name token dominates -> rank 1)
    assert agg["R@1"] == pytest.approx(2.5 / 3)
    assert agg["R@10"] == pytest.approx(1.0)
    assert agg["MRR"] == pytest.approx(1.0)
    assert 0.0 < agg["nDCG@10"] <= 1.0
    assert 0.0 < agg["MAP"] <= 1.0


def test_evaluate_json_schema_and_records(tmp_path):
    evaluate(_config(), results_dir=str(tmp_path))
    doc = json.loads((tmp_path / "metrics" / "bm25_fixture_random.json").read_text("utf-8"))

    prov = doc["provenance"]
    for field in ("config_name", "retriever", "model_id", "seed", "split", "n_examples",
                  "dataset", "accessibility_method", "candidate_scope", "retrieve_k",
                  "timestamp_utc", "comparability_notes"):
        assert field in prov
    assert prov["n_examples"] == 3
    assert prov["retriever"] == "bm25"
    assert prov["dataset"]["name"].startswith("ProofLens mini")   # from fixture metadata.json

    assert set(doc["metrics"]) == {"R@1", "R@10", "MRR", "nDCG@10", "MAP"}
    assert len(doc["examples"]) == 3
    for rec in doc["examples"]:
        assert set(rec) >= {"eid", "gold", "retrieved", "hit_ranks", "metrics"}


def test_evaluate_rank1_is_gold_per_example(tmp_path):
    evaluate(_config(), results_dir=str(tmp_path))
    doc = json.loads((tmp_path / "metrics" / "bm25_fixture_random.json").read_text("utf-8"))
    by_id = {r["eid"]: r for r in doc["examples"]}
    # each example's rank-1 retrieved premise is a gold premise
    for eid in ("continuous_const_fixture#0", "continuous_const_fixture#1", "le_of_lt_fixture#0"):
        rec = by_id[eid]
        assert rec["retrieved"][0] in rec["gold"]


def test_evaluate_respects_accessibility(tmp_path):
    evaluate(_config(), results_dir=str(tmp_path))
    doc = json.loads((tmp_path / "metrics" / "bm25_fixture_random.json").read_text("utf-8"))
    corpus = load_corpus(str(FIX / "corpus.jsonl"))
    # rebuild the per-theorem accessible set and confirm nothing outside it was retrieved
    thm_pos = {
        "continuous_const_fixture#0": ("Mathlib/Topology/Basic.lean", (40, 1)),
        "continuous_const_fixture#1": ("Mathlib/Topology/Basic.lean", (40, 1)),
        "le_of_lt_fixture#0": ("Mathlib/Order/Basic.lean", (30, 1)),
    }
    by_id = {r["eid"]: r for r in doc["examples"]}
    for eid, (path, pos) in thm_pos.items():
        acc = accessible_premises(corpus, path, pos)
        assert set(by_id[eid]["retrieved"]) <= acc


def test_summary_csv_written(tmp_path):
    evaluate(_config(), results_dir=str(tmp_path))
    summary = tmp_path / "metrics" / "summary.csv"
    assert summary.exists()
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["config_name"] == "bm25_fixture" and row["split"] == "random"
    assert float(row["R@10"]) == pytest.approx(1.0)
