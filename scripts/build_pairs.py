"""Build fine-tuning training triplets from a split (Phase 12).

Streams `(query, positive, negatives[])` rows to JSONL + a provenance sidecar `.meta.json`.
Reads `train.json`/`val.json` only (the builder refuses `test.json`). Model-free except for the
optional `bm25s` hard-negative miner (a fast, mining-only BM25).

    # hard negatives (needs bm25s):  pip install bm25s
    python scripts/build_pairs.py --config configs/late_interaction.yaml \
        --split random --split-file train.json --negatives bm25 --n-neg 3 --cap 300
    # random-negative ablation (no bm25s):
    python scripts/build_pairs.py --config configs/late_interaction.yaml \
        --split random --negatives random --n-neg 3 --cap 300
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prooflens.data.corpus import load_corpus
from prooflens.data.pairs import build_triplets
from prooflens.eval.evaluate import _dataset_metadata, _git_commit
from prooflens.utils.logging import get_logger

log = get_logger("build_pairs")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _expand(v):
    return os.path.expandvars(v) if isinstance(v, str) else v


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fine-tuning triplets from a split.")
    ap.add_argument("--config", required=True, help="an LI/BM25 config (for data paths)")
    ap.add_argument("--split", required=True, choices=["random", "novel_premises"])
    ap.add_argument("--split-file", default="train.json",
                    help="train.json (default) or val.json; test.json is refused")
    ap.add_argument("--negatives", default="bm25", choices=["bm25", "random"])
    ap.add_argument("--n-neg", type=int, default=3, help="negatives per (query, positive)")
    ap.add_argument("--cap", type=int, default=None,
                    help="cap per-premise positive occurrences (head-capping); default off")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-n", type=int, default=200, help="bm25: corpus hits retrieved per query")
    ap.add_argument("--neg-skip", type=int, default=2, help="bm25: skip top-N (false-neg guard)")
    ap.add_argument("--neg-window", type=int, default=50, help="bm25: band width to sample from")
    ap.add_argument("--max-theorems", type=int, default=None, help="bound the pass (quick sample)")
    ap.add_argument("--out-dir", default="${SCRATCH}/prooflens/pairs")
    args = ap.parse_args()

    config = load_config(args.config)
    corpus_path = _expand(config["data"]["corpus_path"])
    splits_dir = _expand(config["data"]["splits_dir"])
    split_path = str(Path(splits_dir) / args.split / args.split_file)

    log.info("loading corpus: %s", corpus_path)
    corpus = load_corpus(corpus_path)
    log.info("corpus: %d premises across %d files", len(corpus), len(corpus.paths))
    log.info("building %s triplets from %s (negatives=%s, n_neg=%d, cap=%s) …",
             args.split, split_path, args.negatives, args.n_neg, args.cap)

    out_dir = Path(_expand(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.negatives}_{Path(args.split_file).stem}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    meta_path = out_dir / f"{stem}.meta.json"

    n_rows = 0
    n_neg_total = 0
    n_neg_short = 0                         # rows that got fewer than n_neg negatives
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in build_triplets(
            corpus, split_path,
            negatives=args.negatives, n_neg=args.n_neg, cap=args.cap, seed=args.seed,
            top_n=args.top_n, neg_skip=args.neg_skip, neg_window=args.neg_window,
            max_theorems=args.max_theorems,
        ):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
            n_neg_total += len(row["negatives"])
            if len(row["negatives"]) < args.n_neg:
                n_neg_short += 1
            if n_rows % 20000 == 0:
                log.info("  … %d triplets", n_rows)

    meta = {
        "tool": "scripts/build_pairs.py",
        "split": args.split,
        "split_file": args.split_file,
        "negatives": args.negatives,
        "n_neg": args.n_neg,
        "cap": args.cap,
        "seed": args.seed,
        "bm25_params": ({"top_n": args.top_n, "skip": args.neg_skip, "window": args.neg_window}
                        if args.negatives == "bm25" else None),
        "premise_serialization": "premise_document (full_name + ' ' + code)",
        "n_triplets": n_rows,
        "n_negatives_total": n_neg_total,
        "mean_negatives_per_row": (n_neg_total / n_rows) if n_rows else None,
        "n_rows_under_n_neg": n_neg_short,
        "max_theorems": args.max_theorems,
        "corpus_path": corpus_path,
        "split_path": split_path,
        "dataset": _dataset_metadata(corpus_path),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("wrote %s (%d triplets, mean %.2f negatives/row, %d short)",
             jsonl_path, n_rows, meta["mean_negatives_per_row"] or 0.0, n_neg_short)
    log.info("wrote %s", meta_path)


if __name__ == "__main__":
    main()
