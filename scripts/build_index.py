"""Build a retriever's index from its config (--config). Runs as a Slurm batch job.

Loads the corpus, constructs the configured retriever, and calls build_index(), persisting to the
configured (scratch) index dir. Separated from run_eval so an expensive index build (BM25 tokenise,
dense/LI encode) happens once and is reused across evaluations.
    python scripts/build_index.py --config configs/bm25.yaml [--rebuild]
"""

from __future__ import annotations

import argparse
import os

import yaml

from prooflens.data.corpus import load_corpus
from prooflens.eval.evaluate import build_retriever
from prooflens.utils.logging import get_logger

log = get_logger("build_index")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and persist a retriever index.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    ap.add_argument("--rebuild", action="store_true", help="rebuild even if an index exists")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    corpus_path = os.path.expandvars(config["data"]["corpus_path"])
    log.info("loading corpus: %s", corpus_path)
    corpus = load_corpus(corpus_path)
    log.info("corpus: %d premises across %d files", len(corpus), len(corpus.paths))

    retriever = build_retriever(config)
    log.info("building index for retriever=%s (rebuild=%s)", config["retriever"], args.rebuild)
    retriever.build_index(corpus, rebuild=args.rebuild)
    log.info("index build complete")


if __name__ == "__main__":
    main()
