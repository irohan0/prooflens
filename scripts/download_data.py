"""Stage all downloads on a CSF3 LOGIN NODE (compute nodes are firewalled).

Fetches, onto scratch: (1) LeanDojo Benchmark 4 (Zenodo v10, MD5-verified), (2) the ReProver
ByT5-small retriever checkpoint, (3) the pretrained ColBERT checkpoint for PyLate. Verify presence
before submitting any job; jobs then run offline with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
(docs/CLUSTER.md §2). Model ids were confirmed in Phase 2 (results/phase_logs/phase2.md).

    python scripts/download_data.py --out ~/scratch/prooflens_data           # everything
    python scripts/download_data.py --out ~/scratch/prooflens_data --only benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

BENCHMARK_URL = (
    "https://zenodo.org/records/12740403/files/leandojo_benchmark_4.tar.gz?download=1"
)
BENCHMARK_MD5 = "25e1ee60cd8925b9d2e8673ddcc34b4c"
BENCHMARK_TARBALL = "leandojo_benchmark_4.tar.gz"

# Confirmed HF ids (Phase 2). Keep in sync with the configs.
REPROVER_RETRIEVER = "kaiyuy/leandojo-lean4-retriever-byt5-small"
COLBERT_CHECKPOINT = "lightonai/GTE-ModernColBERT-v1"
# Base for the matched single-vector control — the ModernBERT lineage GTE-ModernColBERT is built on,
# so the LI-vs-single-vector comparison holds the base fixed and varies only the matching mechanism.
SV_BASE = "Alibaba-NLP/gte-modernbert-base"
# ReProver's retrieval-augmented tactic generator (Part 4 / prover loop). It generates the next
# tactic from the proof state PLUS a list of retrieved premises in its context — so we hold this
# generator fixed and vary only which retriever supplies the premises (none / BM25 / FT-SV / FT-LI),
# isolating the retriever's contribution to downstream tactic generation and proof success.
TACGEN = "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_benchmark(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tarball = out / BENCHMARK_TARBALL
    if tarball.exists() and _md5(tarball) == BENCHMARK_MD5:
        print(f"[benchmark] already present and MD5-verified: {tarball}")
    else:
        print(f"[benchmark] downloading {BENCHMARK_URL}")
        req = urllib.request.Request(BENCHMARK_URL, headers={"User-Agent": "prooflens"})
        with urllib.request.urlopen(req) as resp, open(tarball, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        actual = _md5(tarball)
        if actual != BENCHMARK_MD5:
            raise SystemExit(f"[benchmark] MD5 mismatch: {actual} != {BENCHMARK_MD5}")
        print(f"[benchmark] MD5 OK ({actual})")
    print("[benchmark] extracting ...")
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(out)
    print(f"[benchmark] ready under {out / 'leandojo_benchmark_4'}")


def download_hf_model(model_id: str, out: Path) -> None:
    from huggingface_hub import snapshot_download

    dest = out / "models" / model_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[model] downloading {model_id} -> {dest}")
    snapshot_download(repo_id=model_id, local_dir=str(dest))
    print(f"[model] ready: {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage benchmark + model checkpoints onto scratch.")
    ap.add_argument("--out", required=True, help="scratch data root, e.g. ~/scratch/prooflens_data")
    ap.add_argument(
        "--only",
        choices=["benchmark", "reprover", "colbert", "sv_base", "tacgen", "models"],
        help="download only one component (default: everything)",
    )
    args = ap.parse_args()
    out = Path(args.out).expanduser()

    if args.only in (None, "benchmark"):
        download_benchmark(out)
    if args.only in (None, "models", "reprover"):
        download_hf_model(REPROVER_RETRIEVER, out)
    if args.only in (None, "models", "colbert"):
        download_hf_model(COLBERT_CHECKPOINT, out)
    if args.only in (None, "models", "sv_base"):
        download_hf_model(SV_BASE, out)
    if args.only in (None, "models", "tacgen"):
        download_hf_model(TACGEN, out)

    print("\n[done] verify the tree above, then submit jobs with "
          "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.")


if __name__ == "__main__":
    main()
