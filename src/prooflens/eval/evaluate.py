"""Evaluation orchestration: run a retriever over a split and aggregate metrics.

Loads a split, retrieves the top-`retrieve_k` over the **accessible** set per example (the same
accessibility filter for every retriever, matching ReProver's `get_nearest_premises`), averages
R@1 / R@10 / MRR / nDCG@10 / MAP over examples, and writes:
  * `results/metrics/<config>_<split>.json` — provenance header + aggregate + per-example records
    (the full top-`retrieve_k` ranked list is kept so any @k<=retrieve_k metric can be recomputed
    later without re-running retrieval), and
  * a `results/metrics/summary.csv` row (one flat row per run).
Provenance fields follow docs/EVALUATION.md §5 and results/README.md.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from prooflens.data.accessibility import accessible_premises
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split
from prooflens.eval.metrics import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from prooflens.utils.io import write_json
from prooflens.utils.logging import get_logger
from prooflens.utils.seed import set_global_seed

log = get_logger("evaluate")

NDCG_K = 10  # docs/EVALUATION.md fixes nDCG@10


def build_retriever(config: dict):
    """Construct the retriever named by `config['retriever']`. Only BM25 exists in Phase 1's
    first pipeline; dense/late-interaction are added in later phases."""
    rtype = config["retriever"]
    if rtype == "bm25":
        from prooflens.retrievers.bm25 import BM25Retriever

        tok = config.get("tokenizer", {})
        index_dir = config.get("index", {}).get("dir")
        index_dir = os.path.expandvars(index_dir) if index_dir else None
        return BM25Retriever(lowercase=tok.get("lowercase", False), index_dir=index_dir)
    if rtype == "dense":
        from prooflens.retrievers.dense import DenseRetriever

        model = config.get("model", {})
        raw_path = model.get("path")
        model_path = os.path.expandvars(raw_path) if raw_path else model.get("hf_id")
        index_dir = config.get("index", {}).get("dir")
        index_dir = os.path.expandvars(index_dir) if index_dir else None
        return DenseRetriever(
            model_path=model_path,
            max_length=model.get("max_length", 1024),
            batch_size=model.get("batch_size", 64),
            index_dir=index_dir,
            encoder_type=model.get("encoder", "byt5"),
            premise_text=model.get("premise_text", "reprover_serialize"),
        )
    if rtype == "late_interaction":
        from prooflens.retrievers.late_interaction import LateInteractionRetriever

        model = config.get("model", {})
        raw_path = model.get("path")
        model_path = os.path.expandvars(raw_path) if raw_path else model.get("hf_id")
        index_dir = config.get("index", {}).get("dir")
        index_dir = os.path.expandvars(index_dir) if index_dir else None
        return LateInteractionRetriever(
            model_path=model_path,
            query_length=model.get("query_length", 256),
            document_length=model.get("document_length", 300),
            batch_size=model.get("batch_size", 32),
            index_dir=index_dir,
            weighting=config.get("symbol_weighting", {"enabled": False}),
        )
    raise ValueError(f"unknown retriever {rtype!r}")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dataset_metadata(corpus_path: str) -> dict:
    meta_path = Path(corpus_path).with_name("metadata.json")
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        "name": meta.get("dataset_name"),
        "leandojo_version": meta.get("leandojo_version"),
        "commit": meta.get("from_repo", {}).get("commit"),
    }


def _score_example(retriever, accessible, ex, retrieve_k: int, k_list: list[int]) -> dict:
    ranked = [uid for uid, _ in retriever.retrieve(ex.state, accessible, retrieve_k)]
    gold = set(ex.gold)
    metrics = {f"R@{k}": recall_at_k(ranked, gold, k) for k in k_list}
    metrics["MRR"] = reciprocal_rank(ranked, gold)
    metrics[f"nDCG@{NDCG_K}"] = ndcg_at_k(ranked, gold, NDCG_K)
    metrics["MAP"] = average_precision(ranked, gold)
    hit_ranks = {g: (ranked.index(g) + 1 if g in ranked else None) for g in sorted(gold)}
    return {
        "eid": ex.eid,
        "n_accessible": len(accessible),
        "gold": sorted(gold),
        "retrieved": ranked,          # full top-retrieve_k, for later re-scoring at any k
        "hit_ranks": hit_ranks,
        "metrics": metrics,
    }


def _aggregate(records: list[dict], metric_keys: list[str]) -> dict:
    if not records:
        return {k: None for k in metric_keys}
    return {
        k: sum(r["metrics"][k] for r in records) / len(records)
        for k in metric_keys
    }


def _append_summary(results_dir: Path, row: dict) -> None:
    path = results_dir / "metrics" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def evaluate(config: dict, results_dir: str = "results", limit: int | None = None) -> dict:
    """Run the configured retriever over its splits; write JSON + CSV; return {split: metrics}.

    `limit` (optional) evaluates only the first N examples of each split — a quick, backfillable
    subset for sanity/diagnostics. Subset runs are flagged in the provenance header so they are
    never mistaken for the full-corpus number.
    """
    seed = config.get("seed", 42)
    set_global_seed(seed)

    data = config["data"]
    corpus_path = os.path.expandvars(data["corpus_path"])
    splits_dir = os.path.expandvars(data["splits_dir"])
    split_file = data.get("split_file", "test.json")

    log.info("loading corpus: %s", corpus_path)
    corpus = load_corpus(corpus_path)
    log.info("corpus: %d premises across %d files", len(corpus), len(corpus.paths))

    retriever = build_retriever(config)
    log.info("building index for retriever=%s", config["retriever"])
    retriever.build_index(corpus)

    ev = config["eval"]
    k_list = ev.get("k_list", [1, 10])
    retrieve_k = ev.get("retrieve_k", 100)
    splits = ev.get("splits", ["random", "novel_premises"])
    metric_keys = [f"R@{k}" for k in k_list] + ["MRR", f"nDCG@{NDCG_K}", "MAP"]

    dataset_meta = _dataset_metadata(corpus_path)
    git = _git_commit()
    results_path = Path(results_dir)

    out: dict[str, dict] = {}
    for split in splits:
        log.info("evaluating split=%s", split)
        examples = list(load_split(splits_dir, split, corpus, split_file))
        if limit is not None:
            examples = examples[:limit]
        # Accessible sets are large (~58k premises each). load_split yields a theorem's tactics
        # consecutively, so cache only the CURRENT theorem's set (bounded memory, one recompute
        # per theorem) rather than an unbounded dict over all theorems (which would be ~GBs).
        last_key: tuple[str, tuple[int, int]] | None = None
        last_acc: set[str] = set()
        records = []
        for ex in examples:
            key = (ex.file_path, ex.thm_pos)
            if key != last_key:
                last_acc = accessible_premises(corpus, ex.file_path, ex.thm_pos)
                last_key = key
            records.append(_score_example(retriever, last_acc, ex, retrieve_k, k_list))
        agg = _aggregate(records, metric_keys)

        provenance = {
            "config_name": config["name"],
            "retriever": config["retriever"],
            "model_id": config.get("model", {}).get("hf_id", config["retriever"]),
            "seed": seed,
            "split": split,
            "split_file": split_file,
            "n_examples": len(examples),
            "subset_limit": limit,
            "is_full_run": limit is None,
            "dataset": dataset_meta,
            "git_commit": git,
            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "accessibility_method": data.get("accessibility", "computed"),
            "candidate_scope": "accessible (rank all, keep top-k accessible; matches ReProver)",
            "retrieve_k": retrieve_k,
            "metrics_reported": metric_keys,
            # NOTE: dense supports two serializations (Phase 15's matched single-vector control uses
            # full_name+code so its premise text matches the LI/pairs text) — the header must record
            # the one actually used, or the comparability note in the results would be wrong.
            "premise_text": (
                {
                    "reprover_serialize":
                        "ReProver Premise.serialize (code with <a>full_name</a> self-ref markers)",
                    "full_name_code":
                        "full_name + ' ' + code (matched SV control; shared with bm25/LI)",
                }[config.get("model", {}).get("premise_text", "reprover_serialize")]
                if config["retriever"] == "dense" else {
                    "bm25": "full_name + ' ' + code",
                    "late_interaction": "full_name + ' ' + code (shared with bm25)",
                }.get(config["retriever"])
            ),
            "query_text": "proof state (state_before), == ReProver Context.serialize",
            "tokenizer": config.get("tokenizer") if config["retriever"] == "bm25" else None,
            "model_config": (
                config.get("model")
                if config["retriever"] in ("dense", "late_interaction") else None
            ),
            "symbol_weighting": (
                config.get("symbol_weighting")
                if config["retriever"] == "late_interaction" else None
            ),
            "scoring": {
                "bm25": "BM25Okapi over accessible subset",
                "dense": "exact cosine (matmul) over accessible subset",
                "late_interaction": "exact MaxSim over accessible subset (no PLAID ANN)",
            }.get(config["retriever"]),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "comparability_notes": (
                "Candidate set restricted to accessible premises (matches ReProver). "
                "Premise identity = (path, full_name, start). "
                "BM25 tokenizer is a documented design choice; published BM25 is from the LeanDojo "
                "paper (no runnable ReProver BM25)."
            ),
        }

        result = {"provenance": provenance, "metrics": agg, "examples": records}
        out_json = results_path / "metrics" / f"{config['name']}_{split}.json"
        write_json(str(out_json), result)
        log.info("wrote %s (%d examples): %s", out_json, len(examples),
                 {k: (round(v, 4) if v is not None else None) for k, v in agg.items()})

        summary_row = {
            "timestamp_utc": provenance["timestamp_utc"],
            "config_name": config["name"],
            "retriever": config["retriever"],
            "model_id": provenance["model_id"],
            "split": split,
            "n_examples": len(examples),
            **{k: (round(agg[k], 6) if agg[k] is not None else "") for k in metric_keys},
            "seed": seed,
            "git_commit": git or "",
            "job_id": provenance["job_id"],
        }
        _append_summary(results_path, summary_row)
        out[split] = agg

    return out
