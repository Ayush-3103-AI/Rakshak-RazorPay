"""NFR-06 — any rung trains in <= 20 minutes on 4 cores, no GPU.

Asserted against ``train_seconds`` as recorded by the training run itself, not by retraining
here. Retraining inside a perf gate would mean every ``make perf`` spends up to twenty
minutes reproducing a number the training run already measured, and a gate nobody can
afford to run is a gate that gets skipped.

That makes this an assertion about the artifact rather than about the code, and the
honest consequence is stated: it can only fail after someone has trained. It is paired
with a check that the recorded figure is present and plausible, because a missing or zero
``train_seconds`` would pass a naive comparison silently.

Needs T-142 (Lane D, model rungs).
"""

from __future__ import annotations

import json

import pytest
from perf_budgets import MODEL_DIR, assert_budget

TRAIN_BUDGET_S = 20 * 60.0


@pytest.mark.skipif(
    not MODEL_DIR.exists() or not list(MODEL_DIR.glob("*.json")),
    reason=(
        f"no training sidecars in {MODEL_DIR} — NFR-06 needs T-142 (Lane D, model rungs). "
        f"Run `make train RUNG=2` and this asserts for real."
    ),
)
def test_every_rung_trained_inside_the_budget() -> None:
    recorded: dict[str, float] = {}
    for sidecar in sorted(MODEL_DIR.glob("*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        seconds = payload.get("train_seconds")
        assert seconds is not None, (
            f"{sidecar.name} records no train_seconds, so NFR-06 cannot be checked for it. "
            f"A budget with no measurement behind it is the thing this ticket exists to "
            f"retire — have the training path record it."
        )
        assert float(seconds) > 0.0, f"{sidecar.name} records train_seconds={seconds!r}"
        recorded[sidecar.stem] = float(seconds)

    print("\nNFR-06 recorded training times:")
    for name, seconds in sorted(recorded.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {seconds:8.2f} s")
    worst, worst_s = max(recorded.items(), key=lambda kv: kv[1])
    assert_budget("NFR-06", f"slowest rung training run ({worst})", worst_s, TRAIN_BUDGET_S, "s")


@pytest.mark.skipif(
    not MODEL_DIR.exists() or not list(MODEL_DIR.glob("*.json")),
    reason=f"no training sidecars in {MODEL_DIR} — NFR-06 needs T-142 (Lane D).",
)
def test_training_stayed_on_the_declared_core_count() -> None:
    """NFR-06 says 4 cores. A run that quietly used twelve is not the budget being met.

    CLAUDE.md's tech stack is CPU-only by construction, so there is nothing to check about
    a GPU — but the thread count is a real dial and it is the one that would make a
    20-minute budget look met on a machine nobody deploys on.
    """
    for sidecar in sorted(MODEL_DIR.glob("*.json")):
        params = json.loads(sidecar.read_text(encoding="utf-8")).get("hparams", {})
        threads = params.get("num_threads")
        if threads is None:
            continue
        assert threads <= 4, (
            f"{sidecar.name} trained with num_threads={threads}; NFR-06's budget is quoted "
            f"for 4 cores, so a figure measured on more is not a figure against the budget."
        )
