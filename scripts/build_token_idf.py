"""Build the corpus token→IDF table for Phase-22 IDF-weighted late interaction.

Tokenizes every premise's text with the **ColBERT tokenizer** (the same one the query path uses,
so query tokens look up the DF of the identical sub-word), counts document frequencies, and writes
`token_idf.json`. This is a **one-time, split-agnostic** preprocessing step — the table is over
premise *text*, so it serves both the random and novel fine-tuned indices.

Runs on the cluster (needs pylate for the tokenizer); it does not need a GPU — tokenization is
CPU-only — and takes minutes over ~181k premises. Loading the ColBERT model is only to obtain its
exact tokenizer state, guaranteeing the token strings match `encode_query_with_tokens`.

    export DATA_ROOT=$HOME/scratch/prooflens_data/leandojo_benchmark_4
    export MODELS_DIR=$HOME/scratch/prooflens_data/models
    PYTHONPATH=$PWD/src python scripts/build_token_idf.py \
        --model $MODELS_DIR/lightonai__GTE-ModernColBERT-v1 \
        --out $SCRATCH/prooflens/indices/token_idf.json

The `--model` is only a tokenizer source; the *base* ColBERT is fine (fine-tuning does not change
the vocabulary), which keeps the table independent of any single fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import os
import time

from prooflens.data.corpus import load_corpus
from prooflens.retrievers.bm25 import premise_document
from prooflens.retrievers.late_interaction import _ColBERTEncoder
from prooflens.retrievers.token_idf import TokenIDF, document_frequencies, idf_from_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the corpus token→IDF table (Phase 22).")
    ap.add_argument("--corpus", default=None, help="corpus.jsonl (default $DATA_ROOT/corpus.jsonl)")
    ap.add_argument("--model", required=True, help="ColBERT checkpoint dir (tokenizer source)")
    ap.add_argument("--out", required=True, help="output token_idf.json path")
    ap.add_argument("--document-length", type=int, default=300,
                    help="truncate each premise to this many tokens (matches the index encoding)")
    ap.add_argument("--batch", type=int, default=512, help="tokenizer batch size")
    args = ap.parse_args()

    corpus_path = args.corpus or os.path.join(
        os.environ.get("DATA_ROOT", "leandojo_data/leandojo_benchmark_4"), "corpus.jsonl"
    )
    print(f"[idf] loading corpus: {corpus_path}")
    corpus = load_corpus(corpus_path)
    texts = [premise_document(p.full_name, p.code) for p in corpus.all_premises]
    print(f"[idf] {len(texts)} premises")

    # Load ONLY for the tokenizer (device=cpu — no GPU needed for tokenization).
    print(f"[idf] loading ColBERT tokenizer from {args.model}")
    enc = _ColBERTEncoder(args.model, document_length=args.document_length, device="cpu")
    tok = enc.model.tokenizer
    specials = set(enc.special_tokens)

    def doc_tokens():
        """Yield the set of content sub-word token strings for each premise (batched, streamed)."""
        for start in range(0, len(texts), args.batch):
            batch = texts[start : start + args.batch]
            enc_batch = tok(batch, truncation=True, max_length=args.document_length)
            for ids in enc_batch["input_ids"]:
                toks = tok.convert_ids_to_tokens(ids)
                yield [t for t in toks if t not in specials]
            if (start // args.batch) % 50 == 0:
                print(f"[idf]   tokenized {min(start + args.batch, len(texts))}/{len(texts)}")

    t0 = time.time()
    df = document_frequencies(doc_tokens())
    idf = idf_from_df(dict(df), len(texts))
    table = TokenIDF(idf, len(texts), meta={
        "model": args.model,
        "document_length": args.document_length,
        "n_premises": len(texts),
        "n_distinct_tokens": len(idf),
    })
    table.save(args.out)
    elapsed = time.time() - t0

    # A small sanity peek: rarest (highest IDF) and most-common (lowest IDF) content tokens.
    ranked = sorted(idf.items(), key=lambda kv: kv[1])
    print(f"[idf] done in {elapsed:.1f}s — {len(idf)} distinct tokens over {len(texts)} premises")
    print(f"[idf] wrote {args.out}")
    print("[idf] most COMMON tokens (lowest IDF):",
          [t for t, _ in ranked[:8]])
    print("[idf] rarest tokens (highest IDF, ~df=1):",
          [t for t, _ in ranked[-8:]])
    print("[idf] IDF range: "
          f"{ranked[0][1]:.2f} (common) .. {ranked[-1][1]:.2f} (rare)")


if __name__ == "__main__":
    main()
