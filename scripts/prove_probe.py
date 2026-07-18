"""Phase-20 feasibility probe — can we interact with Lean via LeanDojo's `Dojo` on this machine?

This is the make-or-break gate for Part 4 (the prover loop). Before building any proof search, we
must confirm the environment can actually run a tactic against a live Lean proof state. This probe
does the smallest possible end-to-end interaction on the tiny `lean4-example` repo (traces in a few
minutes, unlike mathlib4 which is hours) and reports exactly where it fails if it does.

**Requirements (LeanDojo, per its docs — NOT our main py3.13 venv):**
  - Python 3.9 <= v < 3.12   → a SEPARATE venv (e.g. python3.11)
  - git >= 2.25, wget, elan (the Lean toolchain manager)
  - env GITHUB_ACCESS_TOKEN set to a GitHub personal access token
  - NETWORK at first run (clones + builds the repo); CSF3 compute nodes are firewalled, so run this
    on a LOGIN node first. Whether a *cached* repo lets `Dojo` run offline on a compute node is the
    SECOND thing to test (step B in phase20.md) — that decides if full proof search is viable here.

Usage (on a login node, in the py3.11 lean-dojo venv):
    GITHUB_ACCESS_TOKEN=ghp_xxx python scripts/prove_probe.py

It prints PROBE PASSED / PROBE FAILED and a specific reason. Verify the LeanGitRepo commit and the
`run_tac` return types against the INSTALLED lean-dojo version — the API has shifted across
releases; this targets the documented v4 interface and flags mismatches rather than crashing.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("GITHUB_ACCESS_TOKEN"):
        print("PROBE FAILED: set GITHUB_ACCESS_TOKEN (a GitHub personal access token).")
        print("  LeanDojo needs it to fetch/authenticate the repo. Create one at "
              "github.com/settings/tokens (no scopes needed for public repos).")
        return 2

    try:
        from lean_dojo import Dojo, LeanGitRepo, ProofFinished, Theorem
    except ImportError as e:
        print(f"PROBE FAILED: cannot import lean_dojo ({e}).")
        print("  Install into a Python 3.11 venv:  pip install lean-dojo")
        print("  (LeanDojo requires Python < 3.12 — our main 3.13 venv will NOT work.)")
        return 2

    # The canonical tiny example repo. Pin the commit whose lean-toolchain matches the INSTALLED
    # lean-dojo (both released in lockstep): lean-dojo 4.20.0 <-> lean4-example@main pins
    # leanprover/lean4:v4.20.0. Using a nightly-pinned commit breaks `lake build` (manifest version
    # mismatch). If your lean-dojo is a different version, set the commit to the lean4-example
    # commit whose lean-toolchain names your lean-dojo's Lean version.
    repo = LeanGitRepo(
        "https://github.com/yangky11/lean4-example",
        "7761283d0aed994cd1c7e893786212d2a01d159e",
    )
    # `foo : a + 1 = Nat.succ a` is closed by `rfl` (definitional) -> a clean ProofFinished, which
    # exercises the full run_tac -> proof-closed path, not just interaction.
    theorem = Theorem(repo, "Lean4Example.lean", "foo")
    print(f"[probe] tracing + launching Dojo on {theorem.full_name} "
          f"({repo.url}) — first run clones+builds, a few minutes ...")

    try:
        with Dojo(theorem) as (dojo, init_state):
            print(f"[probe] Dojo launched. initial state:\n{getattr(init_state, 'pp', init_state)}")
            result = dojo.run_tac(init_state, "rfl")     # closes `foo` definitionally
            print(f"[probe] ran `rfl` -> {type(result).__name__}: "
                  f"{getattr(result, 'pp', getattr(result, 'error', result))}")
            if isinstance(result, ProofFinished):
                print("\nPROBE PASSED: Dojo launched, ran a tactic, and CLOSED the goal. "
                      "The Lean interaction works end-to-end.")
            else:
                print("\nPROBE PASSED (interaction works): Dojo launched and executed a tactic and "
                      "returned a state/error. The environment is functional — the specific tactic "
                      "not closing the goal is fine; proof search supplies real tactics.")
        return 0
    except Exception as e:  # noqa: BLE001 — a probe: report ANY failure with its type, don't crash
        print(f"\nPROBE FAILED: {type(e).__name__}: {e}")
        print("  Common causes: elan/Lean not on PATH; no network (run on a LOGIN node first); "
              "GitHub token invalid; toolchain/version mismatch. See phase_logs/phase20.md.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
