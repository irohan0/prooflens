"""Phase-21 orchestration: retrieval-augmented tactic generation, offline.

One run = one **premise condition** on one split. The generator is fixed; the condition decides
what goes into its context:

  * `source: none`            -> no premises (the floor: what the state alone buys you)
  * `source: retrieval_json`  -> the top-`num_retrieved` premises a given retriever already
                                 produced, read straight out of its
                                 `results/metrics/<config>_<split>.json`

Reusing the persisted rankings is the key design choice: `evaluate.py` already stores the full
top-100 UID list per example, so **no retrieval is re-run here** — no BM25 index, no ColBERT, no
GPU-encoded premise matrix, no risk of a retriever behaving differently than it did when it was
measured. The condition is literally the same ranking that produced the published R@k numbers,
which makes the retrieval->generation link airtight rather than approximate.

Alignment is checked, not assumed: the persisted records are matched to freshly-loaded split
examples **by position and verified by `eid`**, and any mismatch raises. Silently mispairing
states with another example's premises would produce plausible-looking numbers that are complete
nonsense, so it fails loudly instead.

Writes `results/metrics/<name>_<split>.json` (provenance + aggregate + per-example records,
including every generated candidate so any match@k can be recomputed without re-running the GPU)
and appends a row to `results/metrics/generation_summary.csv`.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from prooflens.data.audit import short_name, uid_full_name
from prooflens.data.corpus import load_corpus
from prooflens.data.proofs import load_split
from prooflens.eval.tactic_metrics import (
    first_match_rank,
    premise_name_match_at_k,
    tactic_match_at_k,
)
from prooflens.generation.format import (
    format_augmented_state_with_count,
    remove_marks,
    serialize_premise,
)
from prooflens.utils.io import write_json
from prooflens.utils.logging import get_logger
from prooflens.utils.seed import set_global_seed

log = get_logger("generate_eval")


def build_generator(config: dict):
    """Construct the generator named by the config (only ByT5/ReProver exists in Phase 21)."""
    gen = config.get("generator", {})
    gtype = gen.get("type", "byt5")
    if gtype != "byt5":
        raise ValueError(f"unknown generator type {gtype!r}")
    from prooflens.generation.tacgen import (
        DEFAULT_LENGTH_PENALTY,
        DEFAULT_MAX_INP_SEQ_LEN,
        DEFAULT_MAX_OUP_SEQ_LEN,
        ByT5TacticGenerator,
    )

    raw_path = gen.get("path")
    model_path = os.path.expandvars(raw_path) if raw_path else gen.get("hf_id")
    if not model_path:
        raise ValueError("generator needs a `path` or `hf_id`")
    return ByT5TacticGenerator(
        model_path=model_path,
        max_inp_seq_len=gen.get("max_inp_seq_len", DEFAULT_MAX_INP_SEQ_LEN),
        max_oup_seq_len=gen.get("max_oup_seq_len", DEFAULT_MAX_OUP_SEQ_LEN),
        length_penalty=gen.get("length_penalty", DEFAULT_LENGTH_PENALTY),
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_retrieval_records(path: str) -> tuple[dict, list[dict]]:
    """Read a retriever's persisted eval JSON -> (provenance, per-example records)."""
    with open(os.path.expandvars(path), encoding="utf-8") as fh:
        data = json.load(fh)
    if "examples" not in data:
        raise ValueError(f"{path}: not a retrieval results file (no 'examples' key)")
    return data.get("provenance", {}), data["examples"]


def align_retrieved(examples: list, records: list[dict]) -> list[list[str]]:
    """Pair each split example with its persisted top-k UID list, verifying `eid` at every index.

    `evaluate.py` writes records in `load_split` order, so position is the correct join key —
    `eid` alone is not guaranteed unique (it is `full_name#tactic_index`, and theorem full names
    can repeat across files). We therefore join positionally and use `eid` as a checksum: if the
    two sequences ever disagree, the run is aborted rather than silently scoring mismatched pairs.
    """
    if len(examples) != len(records):
        raise ValueError(
            f"retrieval records ({len(records)}) do not match split examples ({len(examples)}). "
            "Was the retrieval run done with a different --limit or split_file?"
        )
    out: list[list[str]] = []
    for i, (ex, rec) in enumerate(zip(examples, records, strict=True)):
        if rec.get("eid") != ex.eid:
            raise ValueError(
                f"alignment broken at index {i}: retrieval record eid={rec.get('eid')!r} "
                f"but split example eid={ex.eid!r}"
            )
        out.append(list(rec.get("retrieved", [])))
    return out


def _premise_texts(corpus, uids: list[str], num_retrieved: int) -> tuple[list[str], int]:
    """Serialize the top-`num_retrieved` UIDs into ReProver premise strings (rank order kept).

    Returns `(texts, n_missing)`. A UID that is not in the corpus is counted and skipped; this
    should never happen (the rankings were produced from this same corpus) so a non-zero count is
    surfaced in the provenance as a corruption signal rather than being swallowed.
    """
    texts: list[str] = []
    missing = 0
    for uid in uids[:num_retrieved]:
        p = corpus.premise_by_uid(uid)
        if p is None:
            missing += 1
            continue
        texts.append(serialize_premise(p.full_name, p.code))
    return texts, missing


def _aggregate(records: list[dict], metric_keys: list[str]) -> dict:
    if not records:
        return {k: None for k in metric_keys}
    return {
        k: sum(r["metrics"][k] for r in records) / len(records)
        for k in metric_keys
    }


def _append_summary(results_dir: Path, row: dict) -> None:
    path = results_dir / "metrics" / "generation_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def generate_eval(
    config: dict,
    results_dir: str = "results",
    limit: int | None = None,
    generator=None,
) -> dict:
    """Run one premise condition on one split; write JSON + CSV; return the aggregate metrics.

    `generator` may be injected (the hermetic tests pass a deterministic fake); otherwise it is
    built from the config. `limit` evaluates only the first N examples — use it for a pilot run
    to measure per-example GPU cost before committing to the full split.
    """
    seed = config.get("seed", 42)
    set_global_seed(seed)

    data = config["data"]
    corpus_path = os.path.expandvars(data["corpus_path"])
    splits_dir = os.path.expandvars(data["splits_dir"])
    split_file = data.get("split_file", "test.json")

    ev = config["eval"]
    split = ev["split"]
    num_samples = ev.get("num_samples", 8)
    k_list = [k for k in ev.get("k_list", [1, num_samples]) if k <= num_samples]
    if 1 not in k_list:
        k_list = [1, *k_list]

    prem = config.get("premises", {})
    source = prem.get("source", "none")
    num_retrieved = prem.get("num_retrieved", 100)

    log.info("loading corpus: %s", corpus_path)
    corpus = load_corpus(corpus_path)
    examples = list(load_split(splits_dir, split, corpus, split_file))
    if limit is not None:
        examples = examples[:limit]
    log.info("split=%s examples=%d condition=%s", split, len(examples), source)

    # -- resolve the premise condition ---------------------------------------------------------
    retrieval_provenance: dict = {}
    if source == "none":
        retrieved_per_example: list[list[str]] = [[] for _ in examples]
    elif source == "retrieval_json":
        results_file = prem.get("retrieval_results")
        if not results_file:
            raise ValueError("premises.source='retrieval_json' needs premises.retrieval_results")
        retrieval_provenance, records = load_retrieval_records(results_file)
        if limit is not None:
            records = records[:limit]
        retrieved_per_example = align_retrieved(examples, records)
    else:
        raise ValueError(f"unknown premises.source {source!r}")

    if generator is None:
        generator = build_generator(config)

    ks = sorted(set(k_list))
    metric_keys = (
        [f"match@{k}" for k in ks]
        + ["match@1_strict"]
        + [f"premise_name@{k}" for k in ks]
    )

    # -- generate ------------------------------------------------------------------------------
    gen_conf = config.get("generator", {})
    max_inp = gen_conf.get("max_inp_seq_len", 2300)
    records_out: list[dict] = []
    total_missing = 0
    t0 = time.time()
    for i, (ex, uids) in enumerate(zip(examples, retrieved_per_example, strict=True)):
        premise_texts, missing = _premise_texts(corpus, uids, num_retrieved)
        total_missing += missing
        # p_drop=0.0: ReProver's datamodule uses `self.p_drop if self.is_train else 0.0`.
        # `n_fitted` is what the model ACTUALLY saw — the byte budget drops most of the offered
        # premises, so reporting len(premise_texts) would overstate retrieval depth ~4x.
        aug_state, n_fitted = format_augmented_state_with_count(
            ex.state, premise_texts, max_len=max_inp, p_drop=0.0
        )
        candidates = generator.generate(aug_state, num_samples)
        tactics = [t for t, _ in candidates]
        scores = [s for _, s in candidates]
        reference = remove_marks(ex.tactic)   # ReProver's generation target
        # Same short-name rule as the Phase-11 audit and Phase-19 lexical stratification, so the
        # premise-name numbers stay comparable with those phases.
        gold_names = {short_name(uid_full_name(u)) for u in ex.gold}

        metrics = {f"match@{k}": tactic_match_at_k(tactics, reference, k) for k in ks}
        metrics["match@1_strict"] = tactic_match_at_k(tactics, reference, 1, normalize=False)
        for k in ks:
            metrics[f"premise_name@{k}"] = premise_name_match_at_k(tactics, gold_names, k)
        records_out.append({
            "eid": ex.eid,
            "n_premises_offered": len(premise_texts),   # what retrieval supplied
            "n_premises_in_context": n_fitted,          # what survived the byte budget
            "reference": reference,
            "gold_premise_names": sorted(gold_names),
            "candidates": tactics,     # full beam, so any match@k is recomputable offline
            "scores": scores,
            "first_match_rank": first_match_rank(tactics, reference),
            "metrics": metrics,
        })
        if (i + 1) % 100 == 0:
            rate = (time.time() - t0) / (i + 1)
            log.info("  %d/%d examples (%.2f s/example, eta %.1f min)",
                     i + 1, len(examples), rate, rate * (len(examples) - i - 1) / 60)
    elapsed = time.time() - t0

    agg = _aggregate(records_out, metric_keys)

    provenance = {
        "config_name": config["name"],
        "task": "retrieval_augmented_tactic_generation",
        "generator_id": gen_conf.get("hf_id"),
        "generator_config": {
            "max_inp_seq_len": max_inp,
            "max_oup_seq_len": gen_conf.get("max_oup_seq_len", 512),
            "length_penalty": gen_conf.get("length_penalty", 0.0),
            "num_samples": num_samples,
        },
        "premise_condition": {
            "source": source,
            "retrieval_results": prem.get("retrieval_results"),
            "num_retrieved_requested": num_retrieved,
            "retriever_config_name": retrieval_provenance.get("config_name"),
            "retriever_type": retrieval_provenance.get("retriever"),
            "retriever_metrics_note": (
                "premises are the PERSISTED top-k from that retrieval run — no retrieval re-run"
            ),
            "uids_missing_from_corpus": total_missing,
        },
        "seed": seed,
        "split": split,
        "split_file": split_file,
        "n_examples": len(examples),
        "subset_limit": limit,
        "is_full_run": limit is None,
        "git_commit": _git_commit(),
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "input_format": (
            "ReProver common.py::format_augmented_state — premises PREPENDED (best-ranked "
            "adjacent to the state), '\\n\\n' separated, UTF-8 BYTE budget = max_inp_seq_len - "
            "len(state), over-budget premises skipped (not break); state last"
        ),
        "premise_text": "ReProver Premise.serialize (code with <a>full_name</a> self-ref markers)",
        "target": "remove_marks(tactic) — ReProver generation datamodule",
        "metrics_reported": metric_keys,
        "metric_definition": (
            "match@k = 1 if any of the top-k generated tactics equals the human tactic after "
            "whitespace normalization. A LOWER BOUND on correctness (textually different tactics "
            "can be semantically valid); comparable ACROSS conditions because the generator and "
            "this metric are identical in every run. NOT a proof-success rate. "
            "premise_name@k = 1 if any of the top-k tactics NAMES a gold premise (short name, "
            "whole-token match via lean_tokenize — the Phase-11/19 rule). That is the signal "
            "premise selection actually controls, and it is the UPPER-bound partner to match@k: "
            "naming the right lemma is necessary but not sufficient for a correct tactic."
        ),
        "premises_in_context_note": (
            "n_premises_offered = what retrieval supplied; n_premises_in_context = how many "
            "survived the UTF-8 byte budget and were actually seen by the model (median ~25 on "
            "real data, so this study tests each retriever's top ~25, not its top 100)."
        ),
        "generation_seconds": elapsed,
        "seconds_per_example": elapsed / len(examples) if examples else None,
        "truncation": (
            generator.truncation_stats() if hasattr(generator, "truncation_stats") else None
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    result = {"provenance": provenance, "metrics": agg, "examples": records_out}
    results_path = Path(results_dir)
    out_json = results_path / "metrics" / f"{config['name']}_{split}.json"
    write_json(str(out_json), result)
    log.info("wrote %s (%d examples): %s", out_json, len(examples),
             {k: (round(v, 4) if v is not None else None) for k, v in agg.items()})

    _append_summary(results_path, {
        "timestamp_utc": provenance["timestamp_utc"],
        "config_name": config["name"],
        "premise_source": source,
        "retriever_config_name": retrieval_provenance.get("config_name", ""),
        "generator_id": gen_conf.get("hf_id", ""),
        "split": split,
        "n_examples": len(examples),
        "num_samples": num_samples,
        **{k: (round(agg[k], 6) if agg[k] is not None else "") for k in metric_keys},
        "seed": seed,
        "git_commit": provenance["git_commit"] or "",
        "job_id": provenance["job_id"],
    })
    return agg
