"""Cluster pre-flight for the tactic generator — run this BEFORE spending GPU-queue time.

The hermetic tests cover everything except the one thing they cannot: that the real ByT5
checkpoint loads and generates sensible Lean tactics through our exact ReProver-replicated call.
This smoke does that on a handful of real examples and prints what the model saw and produced, so
a format mistake is caught by eye in two minutes rather than after a multi-hour full run.

Run on a login node (CPU is fine, just slower) or inside a GPU allocation:

    export MODELS_DIR=$HOME/scratch/prooflens_data/models
    export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
    PYTHONPATH=$PWD/src python scripts/tacgen_smoke.py --n 3

What to check in the output:
  1. `INPUT (tail)` ends with the proof state, and premises appear BEFORE it (ReProver prepends).
  2. Generated candidates look like Lean tactics (`simp`, `exact foo`, `rw [...]`), not gibberish
     — gibberish means the input format or the checkpoint is wrong.
  3. `input bytes` is <= max_inp_seq_len; a high truncation rate means premises are being cut.
"""

from __future__ import annotations

import argparse
import os

from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split
from prooflens.generation.format import (
    format_augmented_state,
    remove_marks,
    serialize_premise,
)
from prooflens.generation.tacgen import ByT5TacticGenerator

TACGEN_DIRNAME = "kaiyuy__leandojo-lean4-retriever-tacgen-byt5-small"


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test the ReProver tactic generator.")
    ap.add_argument("--n", type=int, default=3, help="how many examples to try")
    ap.add_argument("--num-samples", type=int, default=4, help="beam width")
    ap.add_argument("--split", default="random", choices=["random", "novel_premises"])
    ap.add_argument("--num-premises", type=int, default=5,
                    help="how many gold premises to put in context (a stand-in for retrieval)")
    ap.add_argument("--model-path", default=None, help="defaults to $MODELS_DIR/<tacgen dir>")
    args = ap.parse_args()

    model_path = args.model_path or os.path.join(
        os.environ.get("MODELS_DIR", "models"), TACGEN_DIRNAME
    )
    data_root = os.environ.get("DATA_ROOT", "leandojo_data/leandojo_benchmark_4")

    print(f"[smoke] loading corpus from {data_root}")
    corpus = load_corpus(os.path.join(data_root, "corpus.jsonl"))
    examples = list(load_split(data_root, args.split, corpus, "test.json"))[: args.n]
    print(f"[smoke] {len(examples)} examples from split={args.split}")

    print(f"[smoke] loading generator from {model_path}")
    gen = ByT5TacticGenerator(model_path=model_path)
    print(f"[smoke] device={gen.device} max_inp={gen.max_inp_seq_len} "
          f"max_oup={gen.max_oup_seq_len}")

    for ex in examples:
        # Use the example's own gold premises as a stand-in context: if the format is right, the
        # generator should do *well* here, which makes a formatting bug obvious.
        premises = []
        for uid in sorted(ex.gold)[: args.num_premises]:
            p = corpus.premise_by_uid(uid)
            if p is not None:
                premises.append(serialize_premise(p.full_name, p.code))
        aug = format_augmented_state(ex.state, premises, max_len=gen.max_inp_seq_len)
        reference = remove_marks(ex.tactic)

        print("\n" + "=" * 90)
        print(f"eid={ex.eid}  premises_in_context={len(premises)}  "
              f"input bytes={len(aug.encode('utf-8'))}")
        print(f"INPUT (tail):\n...{aug[-400:]}")
        print(f"REFERENCE : {reference!r}")
        for i, (tactic, score) in enumerate(gen.generate(aug, args.num_samples), start=1):
            hit = "  <-- MATCH" if tactic.strip() == reference.strip() else ""
            print(f"  cand {i}: {score:+.4f}  {tactic!r}{hit}")

    print("\n" + "=" * 90)
    print(f"[smoke] truncation: {gen.truncation_stats()}")
    print("[smoke] done — check the three points in this file's docstring before the full run.")


if __name__ == "__main__":
    main()
