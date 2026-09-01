"""Verdict recording and shared dataset helpers for the five parity gates.

08-generator-v2-spec.md §7. Lives beside ``conftest.py`` rather than inside it because
the gate modules import these by name, and ``from conftest import ...`` resolves to
whichever ``conftest`` landed on ``sys.path`` first — there is already one in
``tests/parity/``, so that would be a collision waiting for the wrong afternoon.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from rakshak.generator.config import ScenarioConfig, load_scenario
from rakshak.generator.engine import GeneratedData

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
START = datetime(2026, 1, 1, tzinfo=UTC)

Verdict = Literal["GREEN", "RED", "SKIP"]

#: Small enough that ``make gates`` stays a fast loop; large enough that a per-merchant
#: Fano estimate over the manifest's horizon is stable and every typology has members.
GATE_MERCHANTS = 1_200
GATE_SEED = 20260831

#: The horizon, read from the shipped manifest and never restated in a gate module.
#: T-0112: five gate sites carried a literal ``180`` while the manifest said 365 (RE-FREEZE
#: Amendment 4). Two crashed with IndexError; three silently truncated days 180-364,
#: including the window that holds the confounders G5 exists to measure. One source.
GATE_DAYS: int = load_scenario(CONFIG_PATH).population.n_days


@dataclasses.dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    verdict: Verdict
    statistic: str
    detail: str = ""


#: Module-level on purpose: the terminal-summary hook runs after every test module has
#: been torn down, so the results cannot live on a fixture.
RESULTS: list[GateResult] = []


def record(gate: str, verdict: Verdict, statistic: str, detail: str = "") -> None:
    RESULTS.append(GateResult(gate, verdict, statistic, detail))


def green_if(gate: str, condition: bool, statistic: str, detail: str = "") -> bool:
    """Record a verdict and return the condition, so the caller can assert on it."""
    record(gate, "GREEN" if condition else "RED", statistic, detail)
    return condition


def scenario(**overrides: object) -> ScenarioConfig:
    """The shipped manifest with the gate's population overrides applied.

    The gates read ``configs/scenario_v2.yaml`` rather than a fixture of their own: a
    gate that validated a bespoke configuration would be validating something nothing
    else runs.
    """
    config = load_scenario(CONFIG_PATH)
    population = dataclasses.replace(
        config.population,
        n_merchants=int(overrides.get("n_merchants", GATE_MERCHANTS)),  # type: ignore[call-overload]
        prevalence=float(
            overrides.get("prevalence", config.population.prevalence)  # type: ignore[arg-type]
        ),
    )
    confounders = dataclasses.replace(
        config.confounders, enabled=bool(overrides.get("confounders", True))
    )
    return dataclasses.replace(config, population=population, confounders=confounders)


def daily_counts(
    data: GeneratedData, n_merchants: int = GATE_MERCHANTS, n_days: int = GATE_DAYS
) -> np.ndarray:
    """``(n_merchants, n_days)`` non-refund transaction counts."""
    agg = (
        data.transactions.filter(~pl.col("is_refund"))
        .with_columns(
            m=pl.col("merchant_id").str.slice(1).cast(pl.Int64),
            day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int64),
        )
        .group_by(["m", "day"])
        .len()
    )
    out = np.zeros((n_merchants, n_days), dtype=np.float64)
    out[agg["m"].to_numpy(), agg["day"].to_numpy()] = agg["len"].to_numpy()
    return out


def complete_window_counts(data: GeneratedData, window_days: int) -> pl.DataFrame:
    """``(merchant_id, window, len)`` non-refund counts over COMPLETE windows only.

    Complete windows only. A trailing partial window is a shorter observation period, and
    a count over a shorter period is smaller for a reason that has nothing to do with the
    distribution being tested — it would drag the left tail down and make any comparison
    against an external anchor look worse than the generator deserves.

    Merchant-window cells with no events at all are not emitted. BAF's ``zip_count_4w``
    has a floor of 1 for the same reason: a zip with no applications produces no
    application row to observe it on.

    Shared by G1's ``window_counts`` and G2's ``_per_merchant_mean`` so the window count
    is derived from ``GATE_DAYS`` in one place rather than two.
    """
    return (
        data.transactions.filter(~pl.col("is_refund"))
        .with_columns(
            window=((pl.col("event_time") - START).dt.total_days() // window_days).cast(
                pl.Int64
            )
        )
        .filter(pl.col("window") < GATE_DAYS // window_days)
        .group_by(["merchant_id", "window"])
        .len()
    )
