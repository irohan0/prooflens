"""Render the Phase 1 result figures from results/metrics/summary.csv (never hand-typed numbers).

Produces the five figures in results/README.md §figures:
  1. recall_at_k        — R@1 & R@10 per method, grouped by split (the headline).
  2. generalisation_gap — random vs novel_premises per method (the thesis story).
  3. mrr_comparison     — MRR per method, both splits.
  4. ablation_panel     — ProofLens-LI weighting OFF vs ON.
  5. comparison_table   — the standings (ours vs published) rendered as an image.

All OUR numbers come from summary.csv. Published-literature values are documented citation
constants (they do not drift), used only for the reference rows in the table and the dense
clean-novel bar in the gap plot (the public ReProver checkpoint's own novel number is leaked —
see results/comparison.md — so it is drawn hatched and excluded from the gap). Agg backend so it
runs headless.

    python scripts/plot_results.py --summary results/metrics/summary.csv --out results/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


def _percent_yaxis(ax) -> None:
    """Display fraction-valued ticks (0–1) as percentages (0–100) without relabelling."""
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v * 100:.0f}"))

# -- presentation config ----------------------------------------------------------------------
METHOD_ORDER = ["bm25", "dense_reprover", "late_interaction", "late_interaction_weighted"]
METHOD_LABELS = {
    "bm25": "BM25",
    "dense_reprover": "Dense\nReProver",
    "late_interaction": "LI\n(OFF)",
    "late_interaction_weighted": "LI\n(ON)",
}
SPLIT_ORDER = ["random", "novel_premises"]
SPLIT_LABELS = {"random": "random", "novel_premises": "novel_premises"}
SPLIT_COLORS = {"random": "#4C72B0", "novel_premises": "#DD8452"}
# The one leaked cell (random-trained ReProver checkpoint evaluated on novel_premises-test).
LEAKED = {("dense_reprover", "novel_premises")}

# Published-literature reference numbers (citations, not our measurements). Fractions.
#   LeanDojo 2306.15626 (Lean 3, split-matched checkpoints) + Petrovcic 2510.23637 (Lean 4 random).
PUBLISHED = {
    "BM25 (pub, L3)": {"random": (0.067, 0.172, 0.15), "novel_premises": (0.059, 0.155, 0.14)},
    "ReProver (pub, L4)": {"random": (0.1342, 0.3960, 0.3283)},
    "ReProver (pub, L3)": {"random": (0.135, 0.384, 0.31), "novel_premises": (0.091, 0.276, 0.24)},
}
DENSE_CLEAN_NOVEL_R10 = 0.276  # ReProver clean novel (L3) — reference for the leaked dense cell.


def load_summary(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _latest_by_key(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Keep the last row per (config_name, split) — summary.csv appends over runs."""
    latest: dict[tuple[str, str], dict] = {}
    for r in rows:
        latest[(r["config_name"], r["split"])] = r
    return latest


def _fval(row: dict, key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def _methods_present(latest: dict) -> list[str]:
    present = {k[0] for k in latest}
    return [m for m in METHOD_ORDER if m in present] + sorted(present - set(METHOD_ORDER))


def _annotate(ax, bars, fmt="{:.1f}") -> None:
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h * 100), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                    textcoords="offset points")


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


# -- 1. R@k bars ------------------------------------------------------------------------------
def plot_recall_at_k(latest: dict, out_dir: Path) -> list[Path]:
    methods = _methods_present(latest)
    metrics = ("R@1", "R@10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    width = 0.38
    x = list(range(len(methods)))
    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        top = 0.0
        for si, split in enumerate(SPLIT_ORDER):
            vals = [_fval(latest.get((m, split), {}), metric) or 0.0 for m in methods]
            top = max(top, max(vals))
            pos = [xi + (si - 0.5) * width for xi in x]
            bars = ax.bar(pos, vals, width, label=SPLIT_LABELS[split], color=SPLIT_COLORS[split])
            for m, b in zip(methods, bars, strict=True):        # hatch the leaked cell
                if (m, split) in LEAKED:
                    b.set_hatch("////")
                    b.set_edgecolor("black")
                    ax.annotate("leaked", (b.get_x() + b.get_width() / 2, b.get_height()),
                                ha="center", va="bottom", fontsize=7, color="crimson",
                                xytext=(0, 11), textcoords="offset points", rotation=0)
            _annotate(ax, bars)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
        ax.set_ylim(0, top * 1.25)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"{metric} per method, by split")
        ax.legend(title="split", loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        _percent_yaxis(ax)
    fig.suptitle("Recall@k on LeanDojo Benchmark 4 (Lean 4)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "recall_at_k")


# -- 2. generalisation gap --------------------------------------------------------------------
def plot_generalisation_gap(latest: dict, out_dir: Path) -> list[Path]:
    """random vs novel R@10 per method. The public dense checkpoint's novel is leaked, so we draw
    the published *clean* novel (hatched) — showing the trained model's real drop vs the untrained
    methods' rise."""
    methods = _methods_present(latest)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    x = list(range(len(methods)))
    rnd = [_fval(latest.get((m, "random"), {}), "R@10") or 0.0 for m in methods]
    nov, hatched = [], []
    for m in methods:
        if (m, "novel_premises") in LEAKED:
            nov.append(DENSE_CLEAN_NOVEL_R10)          # substitute published clean-novel
            hatched.append(True)
        else:
            nov.append(_fval(latest.get((m, "novel_premises"), {}), "R@10") or 0.0)
            hatched.append(False)
    b1 = ax.bar([xi - width / 2 for xi in x], rnd, width, label="random",
                color=SPLIT_COLORS["random"])
    b2 = ax.bar([xi + width / 2 for xi in x], nov, width, label="novel_premises",
                color=SPLIT_COLORS["novel_premises"])
    _annotate(ax, b1)
    _annotate(ax, b2)
    for b, h in zip(b2, hatched, strict=True):
        if h:
            b.set_hatch("////")
            b.set_edgecolor("black")
            ax.annotate("published\nclean-novel", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7, color="crimson",
                        xytext=(0, 12), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel("R@10 (%)")
    ax.set_ylim(0, max(max(rnd), max(nov)) * 1.3)
    ax.set_title("Generalisation: random → novel_premises (R@10)\n"
                 "trained dense drops; untrained methods (BM25, LI) do not", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    _percent_yaxis(ax)
    fig.tight_layout()
    return _save(fig, out_dir, "generalisation_gap")


# -- 3. MRR comparison ------------------------------------------------------------------------
def plot_mrr(latest: dict, out_dir: Path) -> list[Path]:
    methods = _methods_present(latest)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.38
    x = list(range(len(methods)))
    top = 0.0
    for si, split in enumerate(SPLIT_ORDER):
        vals = [_fval(latest.get((m, split), {}), "MRR") or 0.0 for m in methods]
        top = max(top, max(vals))
        pos = [xi + (si - 0.5) * width for xi in x]
        bars = ax.bar(pos, vals, width, label=SPLIT_LABELS[split], color=SPLIT_COLORS[split])
        for m, b in zip(methods, bars, strict=True):
            if (m, split) in LEAKED:
                b.set_hatch("////")
                b.set_edgecolor("black")
                ax.annotate("leaked", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7, color="crimson",
                            xytext=(0, 12), textcoords="offset points")
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                        textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylim(0, top * 1.25)
    ax.set_ylabel("MRR")
    ax.set_title("Mean Reciprocal Rank per method, by split", fontweight="bold")
    ax.legend(title="split", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir, "mrr_comparison")


# -- 4. ablation panel (LI OFF vs ON) ---------------------------------------------------------
def plot_ablation(latest: dict, out_dir: Path) -> list[Path]:
    off, on = "late_interaction", "late_interaction_weighted"
    if not any((off, s) in latest for s in SPLIT_ORDER):
        return []
    metrics = ("R@1", "R@10", "MRR")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=False)
    width = 0.38
    x = list(range(len(metrics)))
    for col, split in enumerate(SPLIT_ORDER):
        ax = axes[0][col]
        off_v = [_fval(latest.get((off, split), {}), mm) or 0.0 for mm in metrics]
        on_v = [_fval(latest.get((on, split), {}), mm) or 0.0 for mm in metrics]
        b1 = ax.bar([xi - width / 2 for xi in x], off_v, width, label="weighting OFF",
                    color="#8C8C8C")
        b2 = ax.bar([xi + width / 2 for xi in x], on_v, width, label="weighting ON",
                    color="#55A868")
        for b in (*b1, *b2):
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                        textcoords="offset points")
        for i in range(len(metrics)):                     # % lift arrows
            if off_v[i] > 0:
                lift = 100 * (on_v[i] - off_v[i]) / off_v[i]
                ax.annotate(f"+{lift:.0f}%", (i, max(off_v[i], on_v[i])), ha="center",
                            va="bottom", fontsize=8, color="#2A7F3E", fontweight="bold",
                            xytext=(0, 12), textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, max(max(off_v), max(on_v)) * 1.3)
        ax.set_title(f"split: {SPLIT_LABELS[split]}")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Ablation — symbol-anchored token weighting: OFF vs ON", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "ablation_panel")


# -- 5. comparison table image ----------------------------------------------------------------
def render_comparison_table(latest: dict, out_dir: Path) -> list[Path]:
    def pct(v):
        return "—" if v is None else f"{v * 100:.2f}"

    def dec(v):
        return "—" if v is None else f"{v:.3f}"

    header = ["System", "Type", "R@1", "R@10", "MRR", "nDCG@10"]
    rows: list[list[str]] = []
    label = {"bm25": "BM25 (ours)", "dense_reprover": "Dense ReProver (ours)",
             "late_interaction": "ProofLens-LI OFF (ours)",
             "late_interaction_weighted": "ProofLens-LI ON (ours)"}
    typ = {"bm25": "sparse", "dense_reprover": "dense 1-vec",
           "late_interaction": "late-interaction", "late_interaction_weighted": "late-interaction"}
    for split in SPLIT_ORDER:
        rows.append([f"— split: {split} —", "", "", "", "", ""])
        # published references first
        for name, data in PUBLISHED.items():
            if split in data:
                r1, r10, mrr = data[split]
                rows.append([name, "published", pct(r1), pct(r10), dec(mrr), "—"])
        for m in _methods_present(latest):
            row = latest.get((m, split))
            if not row:
                continue
            flag = " †" if (m, split) in LEAKED else ""
            rows.append([label.get(m, m) + flag, typ.get(m, ""),
                         pct(_fval(row, "R@1")), pct(_fval(row, "R@10")),
                         dec(_fval(row, "MRR")), dec(_fval(row, "nDCG@10"))])

    fig, ax = plt.subplots(figsize=(10.5, 0.36 * (len(rows) + 1) + 0.85))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                   colWidths=[0.28, 0.16, 0.13, 0.13, 0.14, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)
    for j in range(len(header)):                           # header styling
        tbl[0, j].set_facecolor("#33456b")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, start=1):
        tbl[i, 0].set_text_props(ha="left")                # left-align the System column
        if r[0].startswith("— split"):
            for j in range(len(header)):
                tbl[i, j].set_facecolor("#e6e6e6")
                tbl[i, j].set_text_props(fontweight="bold", ha="left")
        elif "(ours)" in r[0]:
            for j in range(len(header)):
                tbl[i, j].set_facecolor("#eef6ee")
            tbl[i, 0].set_text_props(ha="left")
    ax.set_title("ProofLens Phase 1 — standings (ours vs published), LeanDojo Benchmark 4",
                 fontweight="bold", pad=10)
    ax.annotate("R@1/R@10 in %, MRR/nDCG@10 decimals.   † dense novel is cross-split leaked "
                "(not comparable; published clean R@10 = 27.6).", (0.5, 0.04),
                xycoords="figure fraction", ha="center", fontsize=8, color="#444444")
    return _save(fig, out_dir, "comparison_table")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render all Phase 1 figures from summary.csv.")
    ap.add_argument("--summary", default="results/metrics/summary.csv")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    rows = load_summary(args.summary)
    if not rows:
        raise SystemExit(f"no rows in {args.summary}")
    latest = _latest_by_key(rows)
    out = Path(args.out)
    produced = (
        plot_recall_at_k(latest, out)
        + plot_generalisation_gap(latest, out)
        + plot_mrr(latest, out)
        + plot_ablation(latest, out)
        + render_comparison_table(latest, out)
    )
    for p in produced:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
