"""Smoke tests for the plotting helper: each figure renders a PNG + PDF from summary rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_results", SCRIPTS / "plot_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_latest(plot):
    def row(cfg, retr, split, r1, r10, mrr, ndcg):
        return {"config_name": cfg, "retriever": retr, "split": split, "R@1": str(r1),
                "R@10": str(r10), "MRR": str(mrr), "nDCG@10": str(ndcg)}
    rows = []
    for split, scale in (("random", 1.0), ("novel_premises", 1.1)):
        rows += [
            row("bm25", "bm25", split, 0.055 * scale, 0.14 * scale, 0.13 * scale, 0.10 * scale),
            row("dense_reprover", "dense", split, 0.13 * scale, 0.39 * scale, 0.32 * scale, 0.28),
            row("late_interaction", "late_interaction", split, 0.04, 0.10 * scale, 0.10, 0.08),
            row("late_interaction_weighted", "late_interaction", split,
                0.045, 0.11 * scale, 0.11, 0.085),
        ]
    return plot._latest_by_key(rows)


def test_all_five_figures_render(tmp_path):
    plot = _load_plot_module()
    latest = _sample_latest(plot)
    produced = (
        plot.plot_recall_at_k(latest, tmp_path)
        + plot.plot_generalisation_gap(latest, tmp_path)
        + plot.plot_mrr(latest, tmp_path)
        + plot.plot_ablation(latest, tmp_path)
        + plot.render_comparison_table(latest, tmp_path)
    )
    for stem in ("recall_at_k", "generalisation_gap", "mrr_comparison",
                 "ablation_panel", "comparison_table"):
        assert (tmp_path / f"{stem}.png").exists(), stem
        assert (tmp_path / f"{stem}.pdf").exists(), stem
    assert len(produced) == 10                              # 5 figures x (png, pdf)
