"""Phase-21 entrypoint: retrieval-augmented tactic generation from a config.

One run = one premise condition on one split. The generator is fixed; only the premises change.

    # pilot first — measure per-example GPU cost before committing to the full split
    python scripts/run_generate.py --config configs/generate/gen_ft_li_novel.yaml --limit 50

    # the full run
    python scripts/run_generate.py --config configs/generate/gen_ft_li_novel.yaml

Writes results/metrics/<name>_<split>.json and appends to results/metrics/generation_summary.csv.
"""

from __future__ import annotations

import argparse

import yaml

from prooflens.eval.generate_eval import generate_eval


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate retrieval-augmented tactic generation from a YAML config."
    )
    ap.add_argument("--config", required=True, help="path to a configs/generate/*.yaml file")
    ap.add_argument("--results-dir", default="results", help="where to write metrics/")
    ap.add_argument(
        "--split",
        choices=["random", "novel_premises"],
        help="override the config's split (lets each split be its own job)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N examples — use for a pilot to measure s/example before a full run",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="override beam width / number of returned candidates",
    )
    args = ap.parse_args()

    config = load_config(args.config)
    if args.split:
        config.setdefault("eval", {})["split"] = args.split
    if args.num_samples is not None:
        config.setdefault("eval", {})["num_samples"] = args.num_samples
    generate_eval(config, results_dir=args.results_dir, limit=args.limit)


if __name__ == "__main__":
    main()
