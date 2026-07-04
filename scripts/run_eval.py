"""Single evaluation entrypoint: run a retriever over its splits from a config (--config).

Delegates to prooflens.eval.evaluate and writes metrics JSON/CSV to results/metrics/.
    python scripts/run_eval.py --config configs/bm25.yaml
"""

from __future__ import annotations

import argparse

import yaml

from prooflens.eval.evaluate import evaluate


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a retriever from its YAML config.")
    ap.add_argument("--config", required=True, help="path to a configs/*.yaml file")
    ap.add_argument("--results-dir", default="results", help="where to write metrics/ and figures/")
    ap.add_argument(
        "--split",
        choices=["random", "novel_premises"],
        help="evaluate only this split (overrides the config; lets each split be its own job)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N examples per split (quick backfillable diagnostic subset)",
    )
    args = ap.parse_args()

    config = load_config(args.config)
    if args.split:
        config.setdefault("eval", {})["splits"] = [args.split]
    evaluate(config, results_dir=args.results_dir, limit=args.limit)


if __name__ == "__main__":
    main()
