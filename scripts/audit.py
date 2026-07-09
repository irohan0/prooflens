"""Phase 11 — pre-fine-tuning audit & cheap tuning.

Three parts, selected with --part:

  data       model-free data stats over the real benchmark (runs locally; no torch/pylate).
             Retrieval ceiling, pair volume, gold/accessible sizes, premise-frequency (dedup/cap
             decision), gold-name-in-state lexical signal. Streams train.json at bounded memory.

  tokenizer  ModernBERT-tokenizer × Lean-unicode audit + truncation rates + symbol-token fraction.
             Needs the model tokenizer -> run on the cluster.

  sweep      cheap eval-only tuning sweeps that REUSE the cached Phase-8 LI premise index and the
             frozen evaluate(): symbol-weight magnitudes and query_length, on a --limit subset.
             Needs the model + cached index -> run on the cluster.

Examples
    python scripts/audit.py --part data      --config configs/late_interaction.yaml --split random
    python scripts/audit.py --part tokenizer --config configs/late_interaction.yaml
    python scripts/audit.py --part sweep      --config configs/late_interaction.yaml --limit 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from prooflens.data.audit import audit_examples, stream_examples
from prooflens.data.corpus import load_corpus
from prooflens.utils.logging import get_logger

log = get_logger("audit")

# Lean's discriminative unicode glyphs — the tokens ReProver's byte-level ByT5 handles natively and
# a BPE tokenizer may fragment. Used by the tokenizer audit.
LEAN_GLYPHS = list("⊢≤≥→←↔∀∃∘⟨⟩⦃⦄∣∈∉⊆⊂∪∩¬∧∨≠≈≡⁻¹²ℕℤℚℝℂαβγλμ")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _expand(v):
    return os.path.expandvars(v) if isinstance(v, str) else v


def _data_paths(config: dict) -> tuple[str, str, str]:
    data = config["data"]
    return _expand(data["corpus_path"]), _expand(data["splits_dir"]), data.get("split_file")


# -- part: data (local) ---------------------------------------------------------------------------

def run_data(args) -> None:
    config = load_config(args.config)
    corpus_path, splits_dir, cfg_split_file = _data_paths(config)
    split_file = args.split_file or cfg_split_file or "train.json"
    split_path = str(Path(splits_dir) / args.split / split_file)

    log.info("loading corpus: %s", corpus_path)
    corpus = load_corpus(corpus_path)
    log.info("corpus: %d premises across %d files", len(corpus), len(corpus.paths))

    log.info("streaming %s (max_theorems=%s) …", split_path, args.max_theorems)
    examples = stream_examples(split_path, corpus, max_theorems=args.max_theorems)
    with_acc = not args.no_accessibility
    result = audit_examples(examples, corpus if with_acc else None)
    result["_meta"] = {
        "split": args.split,
        "split_file": split_file,
        "max_theorems": args.max_theorems,
        "accessibility_computed": with_acc,
        "corpus_path": corpus_path,
    }

    out = (Path(args.results_dir) / "metrics"
           / f"audit_data_{args.split}_{Path(split_file).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_data_report(result)
    log.info("wrote %s", out)


def _print_data_report(r: dict) -> None:
    pf = r["premise_frequency"]
    lines = [
        "",
        f"=== DATA AUDIT — split={r['_meta']['split']} file={r['_meta']['split_file']} "
        f"(max_theorems={r['_meta']['max_theorems']}) ===",
        f"examples (usable tactics)      : {r['n_examples']:,}",
        f"positive (state,gold) pairs    : {r['n_positive_pairs']:,}",
        f"gold size  mean/median/max     : {_fmt(r['gold_size']['mean'])} / "
        f"{r['gold_size']['median']} / {r['gold_size']['max']}",
        f"gold-name-in-state rate        : {_pct(r['gold_name_in_state_rate'])}"
        "   (lexical signal -> BM25 hard negatives)",
    ]
    if r["accessible_size"]["n"]:
        lines += [
            f"accessible mean/median/p90/max : {_fmt(r['accessible_size']['mean'])} / "
            f"{r['accessible_size']['median']} / {r['accessible_size']['p90']} / "
            f"{r['accessible_size']['max']}",
            f"gold-in-accessible rate        : {_pct(r['gold_in_accessible_rate'])}"
            "   (retrieval ceiling)",
        ]
    lines += [
        f"unique premises used as gold   : {pf['n_unique_gold_premises']:,}",
        f"premise-freq max / median      : {pf['max_count']} / {pf['median_count']}"
        f"   singletons {_pct(pf['singleton_fraction'])}",
        f"head coverage top-1% / top-5%  : {_pct(pf['head_coverage_top1pct'])} / "
        f"{_pct(pf['head_coverage_top5pct'])}   (dedup / capping decision)",
        "most frequent gold premises    : "
        + ", ".join(f"{n}:{c}" for n, c in pf["top_premises"][:8]),
        "",
    ]
    print("\n".join(lines))


def _fmt(x) -> str:
    return f"{x:.1f}" if isinstance(x, (int, float)) else str(x)


def _pct(x) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else str(x)


# -- part: tokenizer (cluster) --------------------------------------------------------------------

def run_tokenizer(args) -> None:
    from transformers import AutoTokenizer  # lazy: cluster-only

    from prooflens.retrievers.late_interaction import is_symbol_subword

    config = load_config(args.config)
    model = config.get("model", {})
    model_path = _expand(model.get("path")) or model.get("hf_id")
    q_len = model.get("query_length", 256)
    d_len = model.get("document_length", 300)
    corpus_path, splits_dir, _ = _data_paths(config)

    log.info("loading tokenizer: %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path)
    unk = tok.unk_token

    # (1) Lean-unicode fragmentation: subwords-per-glyph and any [UNK].
    glyph_rows = []
    n_unk = 0
    for g in LEAN_GLYPHS:
        pieces = tok.tokenize(g)
        if unk in pieces:
            n_unk += 1
        glyph_rows.append((g, len(pieces)))
    mean_sub = sum(n for _, n in glyph_rows) / max(len(glyph_rows), 1)

    # (2) truncation rates + (3) symbol-token fraction, on a sample of real states/premises.
    corpus = load_corpus(corpus_path)
    prem_docs = [_premise_doc(p) for p in corpus.all_premises[: args.sample]]
    split_path = str(Path(splits_dir) / args.split / (args.split_file or "train.json"))
    states = [ex.state for ex in stream_examples(split_path, corpus, max_theorems=args.sample)]
    states = states[: args.sample]

    prem_lens = [len(tok.encode(t, add_special_tokens=True)) for t in prem_docs]
    state_lens = [len(tok.encode(t, add_special_tokens=True)) for t in states]
    specials = frozenset(tok.all_special_tokens)
    n_sym = n_tok = 0
    for t in states[: min(len(states), 500)]:
        for piece in tok.tokenize(t):
            n_tok += 1
            if is_symbol_subword(piece, specials):
                n_sym += 1

    report = {
        "model_path": model_path,
        "query_length": q_len,
        "document_length": d_len,
        "unicode": {
            "n_glyphs": len(glyph_rows),
            "n_glyphs_with_unk": n_unk,
            "mean_subwords_per_glyph": mean_sub,
            "per_glyph": glyph_rows,
        },
        "truncation": {
            "premise_over_doc_len": _frac_over(prem_lens, d_len),
            "premise_len": _len_stats(prem_lens),
            "state_over_query_len": _frac_over(state_lens, q_len),
            "state_len": _len_stats(state_lens),
        },
        "symbol_fraction": (n_sym / n_tok) if n_tok else None,
        "n_premises_sampled": len(prem_docs),
        "n_states_sampled": len(states),
    }
    out = Path(args.results_dir) / "metrics" / "audit_tokenizer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    log.info("wrote %s", out)


def _premise_doc(p):
    from prooflens.retrievers.bm25 import premise_document
    return premise_document(p.full_name, p.code)


def _len_stats(lens: list[int]) -> dict:
    if not lens:
        return {"n": 0}
    s = sorted(lens)
    n = len(s)
    return {"n": n, "mean": sum(s) / n, "median": s[n // 2],
            "p95": s[min(n - 1, int(0.95 * n))], "max": s[-1]}


def _frac_over(lens: list[int], limit: int) -> float | None:
    return (sum(1 for x in lens if x > limit) / len(lens)) if lens else None


# -- part: sweep (cluster) ------------------------------------------------------------------------

def run_sweep(args) -> None:
    """Cheap eval-only sweeps reusing the cached LI premise index + the frozen evaluate().

    Symbol-weight magnitudes and query_length are config overrides; the premise index at
    `index.dir` is unaffected (weighting is query-side; query_length does not change premise
    encoding), so every sweep point reuses the same cached index — no re-indexing.
    """
    from prooflens.eval.evaluate import evaluate

    base = load_config(args.config)
    base.setdefault("eval", {})["splits"] = [args.split]
    audit_dir = str(Path(args.results_dir) / "audit_sweep")

    symbol_weights = [1.5, 2.0, 3.0, 4.0]
    query_lengths = [256, 384, 512]
    rows: list[tuple[str, dict]] = []

    log.info("symbol-weight sweep (weighting ON, cached index) …")
    for sw in symbol_weights:
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"audit_sw{sw}"
        cfg["symbol_weighting"] = {"enabled": True, "symbol_weight": sw, "default_weight": 1.0}
        m = evaluate(cfg, results_dir=audit_dir, limit=args.limit)
        rows.append((f"symbol_weight={sw}", m[args.split]))

    log.info("query-length sweep (weighting OFF, cached index) …")
    for ql in query_lengths:
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"audit_ql{ql}"
        cfg.setdefault("model", {})["query_length"] = ql
        cfg["symbol_weighting"] = {"enabled": False}
        m = evaluate(cfg, results_dir=audit_dir, limit=args.limit)
        rows.append((f"query_length={ql}", m[args.split]))

    print(f"\n=== SWEEP — split={args.split}  (limit={args.limit}) ===")
    print(f"{'variant':22s} {'R@1':>7s} {'R@10':>7s} {'MRR':>7s} {'nDCG@10':>8s}")
    for name, m in rows:
        print(f"{name:22s} {_g(m, 'R@1'):>7} {_g(m, 'R@10'):>7} "
              f"{_g(m, 'MRR'):>7} {_g(m, 'nDCG@10'):>8}")


def _g(m: dict, k: str) -> str:
    v = m.get(k)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


# -- CLI ------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 11 pre-fine-tuning audit.")
    ap.add_argument("--part", required=True, choices=["data", "tokenizer", "sweep"])
    ap.add_argument("--config", required=True, help="an LI config (for data/model paths)")
    ap.add_argument("--split", default="random", choices=["random", "novel_premises"])
    ap.add_argument("--split-file", dest="split_file", default=None,
                    help="override the split file (default: train.json for the audit)")
    ap.add_argument("--max-theorems", type=int, default=None,
                    help="bound the data/tokenizer pass to the first N theorems (fast sample)")
    ap.add_argument("--sample", type=int, default=2000,
                    help="tokenizer part: #premises/#states to sample")
    ap.add_argument("--limit", type=int, default=500,
                    help="sweep part: #examples per eval pass")
    ap.add_argument("--no-accessibility", action="store_true",
                    help="data part: skip the (slower) accessibility stats")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    if args.part == "data":
        run_data(args)
    elif args.part == "tokenizer":
        run_tokenizer(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
