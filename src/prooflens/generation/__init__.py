"""Tactic generation (Phase 21): ReProver's retrieval-augmented generator, offline.

Part 4's ceiling-raiser measures whether better premise selection produces better *proving*.
The full version (best-first proof search against a live Lean via LeanDojo `Dojo`) was gated in
Phase 20 and blocked by a lean-dojo/Lean-4.20 REPL incompatibility (see
`results/phase_logs/phase20.md`). This package implements the planned fallback: hold ReProver's
retrieval-augmented tactic generator FIXED and vary only which retriever supplies its context
premises, then measure next-tactic accuracy against the ground-truth human tactic. No Lean, no
Dojo, no tracing — it runs entirely offline on a GPU node.
"""
