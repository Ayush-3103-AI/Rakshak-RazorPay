"""NFR-05 — a trained model file is <= 20 MB.

An artifact budget rather than a latency one, and it is on ``EvalResult`` via ``PerfBudget``
because charter §2 puts servability inside the success metric: a rung that wins on PR-AUC
and ships a 400 MB artifact has not won.

Needs T-142 (Lane D, model rungs). Every booster on disk is checked, not just Rung 2 — a
budget that only looks at the rung someone remembered to name is a budget with a hole in it.
"""

from __future__ import annotations

import json

import pytest
from perf_budgets import MODEL_DIR, assert_budget

MODEL_SIZE_BUDGET_MB = 20.0


@pytest.mark.skipif(
    not MODEL_DIR.exists() or not list(MODEL_DIR.glob("*.txt")),
    reason=(
        f"no trained boosters in {MODEL_DIR} — NFR-05 needs T-142 (Lane D, model rungs). "
        f"Run `make train RUNG=2` and this asserts for real."
    ),
)
def test_every_trained_model_fits_the_artifact_budget() -> None:
    models = sorted(MODEL_DIR.glob("*.txt"))
    sizes = {path.name: path.stat().st_size / 1_048_576 for path in models}
    print("\nNFR-05 trained artifacts:")
    for name, mb in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} {mb:7.3f} MB")
    worst, worst_mb = max(sizes.items(), key=lambda kv: kv[1])
    assert_budget(
        "NFR-05", f"largest trained model on disk ({worst})", worst_mb, MODEL_SIZE_BUDGET_MB, "MB"
    )


@pytest.mark.skipif(
    not MODEL_DIR.exists() or not list(MODEL_DIR.glob("*.json")),
    reason=f"no training sidecars in {MODEL_DIR} — NFR-05 needs T-142 (Lane D).",
)
def test_the_recorded_size_matches_the_file_on_disk() -> None:
    """``TrainedRung.size_mb`` reads the artifact; the sidecar records what it read.

    If those two ever disagree the recorded number is the one that ends up in the results
    table, and a results table quoting a size nobody can reproduce from the file is exactly
    the provenance problem ``EvalResult`` was built to prevent.
    """
    for sidecar in sorted(MODEL_DIR.glob("*.json")):
        recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("model_size_mb")
        if recorded is None:
            continue
        actual = sidecar.with_suffix(".txt").stat().st_size / 1_048_576
        assert abs(actual - recorded) < 0.01, (
            f"{sidecar.name} records {recorded} MB but the artifact is {actual:.4f} MB"
        )
