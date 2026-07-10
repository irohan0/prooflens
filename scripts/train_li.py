"""Fine-tune the late-interaction (ColBERT) retriever on Lean triplets (Phase 13).

Consumes the Phase-12 pairs JSONL (`{"query","positive","negatives":[...]}`), explodes it to
`(query, positive, negative)` triplets, and fine-tunes a PyLate ColBERT via the ST trainer with the
in-batch + explicit-hard-negative Contrastive loss. Writes a checkpoint the
*unchanged* `LateInteractionRetriever` can load, plus a `training_meta.json` provenance sidecar.

Lengths are pinned to eval (`query_length=384`, `document_length=300` — Phase-11 locked). PyLate /
torch / datasets are imported lazily so this module and its hermetic tests don't require them.

    python scripts/train_li.py --config configs/train/li_ft_random.yaml
    python scripts/train_li.py --config configs/train/li_ft_random.yaml --limit 2000   # quick trial

NOTE: confirm the installed PyLate training API on first cluster use (losses.Contrastive dataset
columns, utils.ColBERTCollator, SentenceTransformerTrainer args); adapt here if the version differs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prooflens.eval.evaluate import _git_commit
from prooflens.utils.logging import get_logger
from prooflens.utils.seed import set_global_seed

log = get_logger("train_li")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _expand(v):
    return os.path.expandvars(v) if isinstance(v, str) else v


# -- pure, testable: pairs JSONL -> (query, positive, negative) triplet records --------------------

def load_triplet_records(path: str, max_samples: int | None = None) -> Iterator[dict]:
    """Yield exploded `{"query","positive","negative"}` rows from a Phase-12 pairs JSONL.

    Each source row `{"query","positive","negatives":[n1,n2,...]}` yields one triplet per negative
    (single-negative rows are the universally-compatible contrastive format; in-batch negatives add
    the rest at train time). Rows with no negatives are skipped (no triplet). `max_samples` bounds
    the number of emitted triplets (the `--limit` trial).
    """
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query, positive = row["query"], row["positive"]
            for neg in row.get("negatives", []):
                yield {"query": query, "positive": positive, "negative": neg}
                n += 1
                if max_samples is not None and n >= max_samples:
                    return


def _read_meta(pairs_path: str) -> dict | None:
    """The Phase-12 provenance sidecar written next to the pairs JSONL, if present."""
    p = Path(pairs_path).parent / (Path(pairs_path).stem + ".meta.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# -- training (lazy heavy imports) ----------------------------------------------------------------

def train(config: dict, limit: int | None = None) -> str:
    tr = config["train"]
    if tr.get("use_cpu", False):
        # Hide CUDA BEFORE torch is imported: on Transformers v5+ the trainer still grabs a visible
        # GPU even with use_cpu=True, which fails on a busy login-node GPU. This is torch's own
        # recommended way to force CPU. Real GPU runs leave use_cpu unset -> CUDA stays visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from datasets import Dataset
    from pylate import losses, models, utils
    from sentence_transformers import (
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )

    base = config["base_model"]
    seed = tr.get("seed", 42)
    set_global_seed(seed)

    base_path = _expand(base.get("path")) or base.get("hf_id")
    query_length = base.get("query_length", 384)
    document_length = base.get("document_length", 300)
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

    # Force the model onto CPU for the login-node API-confirmation trial: PyLate's ColBERT moves the
    # model to CUDA at construction (before the trainer's use_cpu applies), which fails on a busy
    # login-node GPU. On a real GPU allocation use_cpu is unset -> device=None -> auto-detect CUDA.
    device = "cpu" if tr.get("use_cpu", False) else None
    log.info("loading base ColBERT: %s (q=%d, d=%d, device=%s)",
             base_path, query_length, document_length, device or "auto")
    model = models.ColBERT(
        model_name_or_path=base_path,
        query_length=query_length,
        document_length=document_length,
        device=device,
    )
    loss = losses.Contrastive(model=model)

    do_eval = eval_ds is not None
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=tr.get("epochs", 1),
        per_device_train_batch_size=tr.get("batch_size", 32),
        per_device_eval_batch_size=tr.get("batch_size", 32),
        learning_rate=float(tr.get("lr", 3.0e-6)),
        warmup_ratio=tr.get("warmup_ratio", 0.05),
        bf16=tr.get("bf16", True),
        fp16=tr.get("fp16", False),
        use_cpu=tr.get("use_cpu", False),   # CPU API-confirmation trial (login node, no GPU needed)
        seed=seed,
        logging_steps=tr.get("logging_steps", 100),
        save_strategy="steps" if do_eval else "epoch",
        save_steps=tr.get("save_steps", 500),
        eval_strategy="steps" if do_eval else "no",
        eval_steps=tr.get("eval_steps", 500),
        save_total_limit=2,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    log.info("training … (epochs=%s, batch=%s, lr=%s, bf16=%s)",
             tr.get("epochs", 1), tr.get("batch_size", 32), tr.get("lr", 3.0e-6),
             tr.get("bf16", True))
    trainer.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    meta = {
        "tool": "scripts/train_li.py",
        "config_name": config.get("name"),
        "base_model": base_path,
        "query_length": query_length,
        "document_length": document_length,
        "pairs": pairs_path,
        "pairs_meta": _read_meta(pairs_path),
        "val_pairs": val_path,
        "n_train_triplets": len(train_records),
        "n_val_triplets": n_val,
        "epochs": tr.get("epochs", 1),
        "batch_size": tr.get("batch_size", 32),
        "lr": float(tr.get("lr", 3.0e-6)),
        "warmup_ratio": tr.get("warmup_ratio", 0.05),
        "seed": seed,
        "limit": limit,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(output_dir) / "training_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    log.info("saved fine-tuned ColBERT + training_meta.json -> %s", output_dir)
    return output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the late-interaction retriever on Lean.")
    ap.add_argument("--config", required=True, help="a configs/train/*.yaml file")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N triplets (quick trial to right-size epochs/LR)")
    args = ap.parse_args()
    train(load_config(args.config), limit=args.limit)


if __name__ == "__main__":
    main()
