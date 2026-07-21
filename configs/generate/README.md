# configs/generate — Phase 21 (retrieval-augmented tactic generation)

One config = **one premise condition on one split**. The generator
(`kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small`) is **identical in every file**; the only
thing that varies is `premises:`. That is the whole experimental design — any difference in
next-tactic accuracy is attributable to the retriever, because nothing else moves.

| condition | `premises.source` | what it answers |
|---|---|---|
| `gen_none_*` | `none` | the floor: what the proof state alone buys the generator |
| `gen_bm25_*` | BM25's persisted top-k | does a sparse lexical retriever help downstream? |
| `gen_ft_sv_*` | the matched single-vector control's top-k | the controlled single-vector comparator |
| `gen_ft_li_*` | fine-tuned late interaction (weighting ON) | our contribution |

`premises.retrieval_results` points at the **already-persisted** `results/metrics/*.json` from the
retrieval runs, so no retrieval is re-executed — the premises are literally the ranking that
produced the published R@k numbers. Paths are relative to the repo root, so jobs must `cd` there
(`slurm/generate_eval.sh` does).

**The headline comparison is on `novel_premises`**, where the retrievers actually differ
(Phase 15: FT-SV drops 18% on novel while FT-LI stays flat). `random` is run as the contrast case.

Run a `--limit 50` pilot first to measure seconds/example before committing to a full split —
the generator is byte-level ByT5 over a ~2300-byte context, so it is not cheap.
