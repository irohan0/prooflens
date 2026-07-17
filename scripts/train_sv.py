"""Fine-tune the MATCHED single-vector control on the same Lean triplets (Phase 15).

The single-vector counterpart to `scripts/train_li.py`: it consumes the **identical** Phase-12
triplets and trains with the **same budget** and the **same base lineage** (`gte-modernbert-base`,
the ModernBERT that GTE-ModernColBERT is built on), but pooled to **one vector per text** — a
`SentenceTransformer` + `MultipleNegativesRankingLoss` (in-batch + the explicit hard negative) —
instead of ColBERT MaxSim. The pooling is whatever the checkpoint's own ST config defines
(**gte-modernbert-base uses CLS-token pooling** — verified from the module stack in the Phase-15
trial run, job 17639680 — not mean; either way the text is bottlenecked through a single 768-d
vector, which is the thing under test). So the ONLY difference vs the LI run is the matching
mechanism (single-vector pooling vs multi-vector late interaction) → it isolates whether *late
interaction*, not just fine-tuning, is what shrinks the random→novel gap.

Evaluated through the SAME frozen harness via `dense.py`'s `sentence_transformer` encoder option
(exact cosine over the accessible set) — identical metrics/accessibility to every other retriever.

    python scripts/train_sv.py --config configs/train/sv_ft_random.yaml
    python scripts/train_sv.py --config configs/train/sv_ft_random.yaml --limit 10000   # trial

torch / sentence-transformers / datasets are imported lazily (cluster-only).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prooflens.data.pairs import load_triplet_records
from prooflens.eval.evaluate import _git_commit
from prooflens.utils.logging import get_logger
from prooflens.utils.seed import set_global_seed

log = get_logger("train_sv")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _expand(v):
    return os.path.expandvars(v) if isinstance(v, str) else v


def _read_meta(pairs_path: str) -> dict | None:
    p = Path(pairs_path).parent / (Path(pairs_path).stem + ".meta.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def train(config: dict, limit: int | None = None) -> str:
    tr = config["train"]
    if tr.get("use_cpu", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # force CPU before torch import (login trial)

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )

    base = config["base_model"]
    seed = tr.get("seed", 42)
    set_global_seed(seed)

    base_path = _expand(base.get("path")) or base.get("hf_id")
    max_length = base.get("max_length", 512)
    output_dir = _expand(config["output_dir"])
    pairs_path = _expand(tr["pairs"])
    val_path = _expand(tr.get("val_pairs")) if tr.get("val_pairs") else None

    log.info("loading triplets: %s", pairs_path)
    train_records = list(load_triplet_records(pairs_path, max_samples=limit))
    if not train_records:
        raise SystemExit(f"no training triplets in {pairs_path}")
    train_ds = Dataset.from_list(train_records)
    log.info("train triplets: %d (from %s)", len(train_records), Path(pairs_path).name)

    eval_ds = None
    n_val = 0
    if val_path and Path(val_path).exists():
        val_records = list(load_triplet_records(val_path, max_samples=(limit or 0) or None))
        if val_records:
            eval_ds = Dataset.from_list(val_records)
            n_val = len(val_records)
            log.info("val triplets: %d (from %s)", n_val, Path(val_path).name)

    device = "cpu" if tr.get("use_cpu", False) else None
    log.info("loading base single-vector model: %s (max_len=%d, device=%s)",
             base_path, max_length, device or "auto")
    model = SentenceTransformer(base_path, device=device)
    model.max_seq_length = max_length
    # Log the module stack (Transformer -> Pooling -> ...) so the run record shows the checkpoint's
    # OWN pooling mode (cls vs mean) — the control uses whatever gte-modernbert-base defines, and
    # the report must state what that actually was, not assume.
    log.info("single-vector module stack: %s", model)
    loss = losses.MultipleNegativesRankingLoss(model)      # in-batch + the explicit hard negative

    do_eval = eval_ds is not None
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr.get("epochs", 1),
        per_device_train_batch_size=tr.get("batch_size", 32),
        per_device_eval_batch_size=tr.get("batch_size", 32),
        learning_rate=float(tr.get("lr", 2.0e-5)),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        bf16=tr.get("bf16", True),
        fp16=tr.get("fp16", False),
        use_cpu=tr.get("use_cpu", False),
        seed=seed,
        logging_steps=tr.get("logging_steps", 500),
        save_strategy="steps" if do_eval else "epoch",
        save_steps=tr.get("save_steps", 5000),
        eval_strategy="steps" if do_eval else "no",
        eval_steps=tr.get("eval_steps", 5000),
        save_total_limit=2,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, loss=loss,
    )
    log.info("training single-vector control … (epochs=%s, batch=%s, lr=%s)",
             tr.get("epochs", 1), tr.get("batch_size", 32), tr.get("lr", 2.0e-5))
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    meta = {
        "tool": "scripts/train_sv.py",
        "config_name": config.get("name"),
        "architecture": ("single-vector (SentenceTransformer, checkpoint's own pooling — "
                         "gte-modernbert-base = CLS token, MultipleNegativesRanking)"),
        "base_model": base_path,
        "max_length": max_length,
        "pairs": pairs_path,
        "pairs_meta": _read_meta(pairs_path),
        "val_pairs": val_path,
        "n_train_triplets": len(train_records),
        "n_val_triplets": n_val,
        "epochs": tr.get("epochs", 1),
        "batch_size": tr.get("batch_size", 32),
        "lr": float(tr.get("lr", 2.0e-5)),
        "warmup_ratio": tr.get("warmup_ratio", 0.05),
        "seed": seed,
        "limit": limit,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(output_dir) / "training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info("saved fine-tuned single-vector model + training_meta.json -> %s", output_dir)
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the matched single-vector control on Lean.")
    ap.add_argument("--config", required=True, help="a configs/train/sv_*.yaml file")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N triplets (trial)")
    args = ap.parse_args()
    train(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
