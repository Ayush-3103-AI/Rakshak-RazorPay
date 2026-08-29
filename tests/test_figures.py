"""FR-020's figure renders from the committed sweep CSV (T-0007b, figure assigned 2026-08-29).

One check, deliberately: the figure computes nothing of its own, so the only thing
that can break is the draw. Everything it plots is already pinned by
`tests/test_policy.py` against the frame it comes from.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rakshak.config import RESULTS_DIR
from rakshak.eval.figures import render_sensitivity_figure

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_figure_renders_from_the_committed_sweep_csv(tmp_path):
    """`make figures` must work off `results/sensitivity.csv` with no model refit."""
    csv_path = RESULTS_DIR / "sensitivity.csv"
    if not csv_path.exists():
        pytest.skip(f"{csv_path} not generated yet - run `make eval` first")

    frame = pd.read_csv(csv_path)
    # The three panels read these columns; a rename upstream must fail here, loudly.
    for column in ("asymmetry", "model", "savings", "margin_abs", "margin_rel",
                   "hold_threshold_median", "hold_threshold_median_at_risk"):
        assert column in frame.columns, f"sweep CSV lost the {column!r} column"

    out = render_sensitivity_figure(frame, tmp_path / "figures" / "sensitivity.png")
    assert out.exists()
    assert out.read_bytes()[:8] == _PNG_MAGIC
    assert out.stat().st_size > 10_000, "figure is suspiciously small - did a panel fail?"
