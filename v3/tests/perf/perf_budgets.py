"""Measurement machinery for the perf budgets (NFR-01..06, NFR-10).

**A number in a doc is a wish; a number in an assert is a budget.** That is the whole of
T-150 and the reason this directory exists: every NFR in 06-requirements-v2.md §C that
carries a number is asserted here and fails CI when it is violated.

Three rules the suite holds itself to, because a flaky timing test is worse than no timing
test — it trains everyone to re-run until green, which is the same as deleting it:

1. **Warm up, then measure many.** Every measurement discards a warmup pass and then times
   ``iterations`` calls, repeated over ``batches``. A single sample measures the scheduler.
2. **Report the whole distribution, assert on one number.** Median, best-batch p99 and
   worst-batch p99 are printed for every budget along with the margin, so a run that passes
   narrowly says so out loud rather than looking identical to one that passes by 20x. The
   assertion is on the best batch's p99: this tree is built with several agents running
   concurrently on one machine, and the minimum across batches is the estimate least
   contaminated by whatever else held the core. Stated here rather than buried, because it
   is the one methodological choice in the suite a reviewer should want to see argued.
3. **Never widen a budget to go green.** If a measurement does not fit, the number goes in
   LIMITATIONS.md. NFR-04 is exactly that case today; see ``test_state_size.py``.

This lives in a *named module* rather than in ``conftest.py`` on purpose. ``pythonpath``
puts ``tests/parity`` on the path, so a bare ``import conftest`` from anywhere in the tree
resolves to *that* conftest — the parity suite's — and a second suite importing its own by
that name breaks collection for both the moment they are collected together. The parity
suite solved this with ``parity_harness.py``; this is the same solution with a different
name.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where Lane D's `make train` puts boosters. Every model-dependent budget is gated on files
#: here existing, and skips with the dependency named when they do not.
MODEL_DIR = ROOT / "data" / "v2" / "models"

#: Long enough that every trailing window in the register is full: the widest is
#: `v_declared_ratio`'s 30 days, on top of a 30-day warmup. A state measured before its rings
#: fill is a state measured below its steady size, which is the flattering number.
STREAM_DAYS = 75
SEED = 42

#: Accumulated by `assert_budget`, written out at session end by the conftest hook. This is
#: what `eval.metrics.PerfBudget` asks for by refusing to invent its own numbers.
MEASURED: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class Timing:
    """One measurement, in milliseconds."""

    median_ms: float
    p99_ms: float
    p99_worst_ms: float
    iterations: int
    batches: int

    def line(self) -> str:
        return (
            f"median {self.median_ms:.4f} ms · p99 {self.p99_ms:.4f} ms "
            f"(worst batch {self.p99_worst_ms:.4f} ms) "
            f"· {self.batches}x{self.iterations} iterations · {os.cpu_count()} cores visible"
        )


def measure(
    fn: Callable[[], object], *, iterations: int = 500, batches: int = 5, warmup: int = 100
) -> Timing:
    """Time ``fn`` honestly. See rules 1 and 2 in the module docstring."""
    for _ in range(warmup):
        fn()
    p99s: list[float] = []
    medians: list[float] = []
    for _ in range(batches):
        samples = np.empty(iterations, dtype=np.float64)
        for i in range(iterations):
            started = time.perf_counter_ns()
            fn()
            samples[i] = time.perf_counter_ns() - started
        p99s.append(float(np.percentile(samples, 99)) / 1e6)
        medians.append(float(np.median(samples)) / 1e6)
    return Timing(
        median_ms=float(np.median(medians)),
        p99_ms=min(p99s),
        p99_worst_ms=max(p99s),
        iterations=iterations,
        batches=batches,
    )


def assert_budget(nfr: str, what: str, measured: float, budget: float, unit: str) -> None:
    """Record the measurement, print it with its margin, and fail if it is over.

    The margin is printed on success too. A budget met by 1% and a budget met by 2000x are
    the same green tick and very different facts, and the second number is the one that says
    whether the next feature fits.
    """
    MEASURED[nfr] = {
        "what": what,
        "measured": measured,
        "budget": budget,
        "unit": unit,
        "cores_visible": os.cpu_count(),
        "passed": measured <= budget,
    }
    margin = budget / measured if measured > 0 else float("inf")
    print(f"\n{nfr}  {what}: {measured:.4f} {unit} against {budget} {unit} — {margin:.2f}x margin")
    assert measured <= budget, (
        f"{nfr} VIOLATED. {what} measured {measured:.4f} {unit} against a budget of "
        f"{budget} {unit} ({measured / budget:.2f}x over). Do not raise the budget to make "
        f"this pass — the number is what makes the claim checkable. Either make it fit or "
        f"write the measurement into LIMITATIONS.md and say so in the report."
    )


def write_measurements() -> Path | None:
    if not MEASURED:
        return None
    out = ROOT / "data" / "v2" / "perf" / "budgets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(MEASURED, indent=2, sort_keys=True), encoding="utf-8")
    return out


def rung_artifacts(rung: int = 2, seed: int = SEED) -> tuple[Path, Path]:
    """The booster and its sidecar for a trained rung, whether or not they exist yet."""
    return MODEL_DIR / f"rung{rung}_seed{seed}.txt", MODEL_DIR / f"rung{rung}_seed{seed}.json"


#: T-150 depends on T-142, and Lane D is building the rungs. Everything that needs a trained
#: booster carries this: it skips with the dependency named — never a silent pass — and
#: asserts for real the moment `make train RUNG=2` has written one.
needs_trained_rung = pytest.mark.skipif(
    not rung_artifacts()[0].exists(),
    reason=(
        f"no trained booster at {rung_artifacts()[0]} — this budget needs T-142 (Lane D, "
        f"model rungs). Run `make train RUNG=2` and it asserts for real."
    ),
)
