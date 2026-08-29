"""FR-020's figure: the cost-asymmetry sweep, drawn.

`06-requirements.md` FR-020 requires `results/sensitivity.md` to report the sweep
"as a table AND a figure". T-0010 owned `results/figures/` and was cut in the
2026-08-28 re-plan, which left the figure clause with no owner; it was assigned
here on 2026-08-29 rather than struck.

The figure is a rendering of `results/sensitivity.csv` and adds no number of its
own. That is deliberate: a figure that computes is a second implementation that
can disagree with the table beside it, which is the same argument that made
T-0014 a read-only viewer. Re-running `--figures-only` redraws from the committed
CSV without refitting a model.

Three panels, in the order a reader needs them:

1. **Absolute savings by asymmetry, every model including `random`.** The `random`
   floor is on this panel because omitting it invites exactly the misreading
   panel 2 would otherwise produce — most of the savings *level* is the cost
   matrix, not detection (`07-math.md` §6, AP-06).
2. **Relative margin of the proposal over `rules`** — FR-020(a) — with NFR-001's
   +20% bar and the crossing marked, which is FR-020(b).
3. **Optimal hold threshold across the sweep** — FR-020(d) — both the at-risk
   median and the degenerate all-merchant median, because the degeneracy is the
   finding.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display on CI or a laptop running headless.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = ["render_sensitivity_figure"]

# Fixed per-model colours so the figure reads the same across regenerations.
_COLOURS = {
    "random": "#999999",
    "rules": "#1f77b4",
    "gbdt": "#ff7f0e",
    "hmm": "#d62728",
}
_MARGIN_BAR = 0.20  # NFR-001's >=20% relative improvement over the rule engine.


def render_sensitivity_figure(
    frame: pd.DataFrame,
    out_path: Path,
    proposal_model: str = "hmm",
    reference_model: str = "rules",
) -> Path:
    """Draw the FR-020 sweep figure from a `sweep_cost_asymmetry` frame.

    Args:
        frame: Tidy sweep frame, one row per (asymmetry, model).
        out_path: PNG destination. Parent directories are created.
        proposal_model: The model whose margin panel 2 plots.
        reference_model: The floor that margin is measured against.

    Returns:
        `out_path`, for the caller to log.
    """
    points = frame.drop_duplicates("asymmetry").sort_values("asymmetry")
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 11.0), sharex=True)

    # --- Panel 1: absolute savings, every model -----------------------------
    ax = axes[0]
    for model, block in frame.groupby("model", sort=False):
        block = block.sort_values("asymmetry")
        ax.plot(
            block["asymmetry"],
            block["savings"],
            marker="o",
            markersize=3.5,
            label=model,
            color=_COLOURS.get(str(model)),
            linewidth=2.0 if model == proposal_model else 1.4,
        )
    ax.set_ylabel("savings (fraction of loss averted)")
    ax.set_title(
        "Savings by cost asymmetry — note how close `random` runs to `rules`:\n"
        "most of the savings LEVEL is the cost matrix, not detection (AP-06)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # --- Panel 2: relative margin, the bar, and the crossing (FR-020 a, b) --
    ax = axes[1]
    ax.plot(points["asymmetry"], points["margin_rel"], marker="o", markersize=3.5,
            color=_COLOURS.get(proposal_model), linewidth=2.0)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.axhline(_MARGIN_BAR, color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"NFR-001 bar (+{_MARGIN_BAR:.0%})")
    positive = points[points["margin_abs"] > 0.0]
    if not positive.empty:
        crossing = float(positive["asymmetry"].iloc[0])
        below = points[points["margin_abs"] <= 0.0]["asymmetry"]
        lower = float(below.max()) if not below.empty else crossing
        ax.axvspan(points["asymmetry"].min(), lower, color="#d62728", alpha=0.07)
        ax.annotate(
            f"margin crosses zero\nbetween {lower:.1f} and {crossing:.1f}",
            xy=(crossing, 0.0), xytext=(crossing, float(points["margin_rel"].max()) * 0.45),
            fontsize=8, ha="left",
            arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 0.9},
        )
    ax.set_ylabel(f"relative margin, {proposal_model} over {reference_model}")
    ax.set_title(
        f"FR-020(a)+(b): the shaded band is where {proposal_model} LOSES to "
        f"{reference_model}.\nNo verdict here — K2 is T-0011's, on `test`.",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    # --- Panel 3: optimal thresholds (FR-020 d) -----------------------------
    ax = axes[2]
    ax.plot(points["asymmetry"], points["hold_threshold_median_at_risk"], marker="o",
            markersize=3.5, color="#1f77b4", label="median p* over at-risk merchants")
    ax.plot(points["asymmetry"], points["hold_threshold_median"], marker="s",
            markersize=3.5, color="#999999", linestyle=":",
            label="median p* over ALL merchants (degenerate at 1.0)")
    ax.set_ylabel("optimal hold threshold p*")
    ax.set_xlabel("FP cost per INR 100 of realised fraud loss (log scale)")
    ax.set_title(
        "FR-020(d): both medians ship. The all-merchant median pins at 1.0\n"
        "because most merchants have L_m = 0 — the degeneracy is the finding.",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)

    for ax in axes:
        ax.set_xscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, metadata={"Software": None, "Creation Time": None})
    plt.close(fig)
    return out_path
