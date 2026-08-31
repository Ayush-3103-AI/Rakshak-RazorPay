"""T-114: the platform confounder layer P1-P6 moves everyone, and nothing can see it.

08-generator-v2-spec.md §4. This is the v2 contribution: platform-wide events that shift
every merchant's features with zero fraud occurring. Gate G5 asks whether a detector can
tell them apart from adversarial drift, and that question is only meaningful if:

1. the layer is genuinely **independent** — no persona or typology code can read it, so
   "the confounder is a separate layer" is a property of the program rather than a claim
   about the author's intent;
2. the effect is genuinely **large** — population mean |z| above 1.0 inside every event
   window, or the null test is passed by an effect too small to have tested anything;
3. the effect is genuinely **heterogeneous** — P1 must hit L2 harder than L4, or the
   cohort residual has an unfairly easy job and G5 passes for the wrong reason.

All three are asserted here. The run is at ``prevalence = 0``: every merchant is
legitimate, so anything that moves is the platform.
"""

from __future__ import annotations

import ast
import dataclasses
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import rakshak.generator.confounders as confounders_mod
import rakshak.generator.personas as personas_mod
import rakshak.generator.typologies as typologies_mod
from rakshak.generator.config import ScenarioConfig, load_scenario
from rakshak.generator.confounders import (
    ConfounderWindow,
    build_layer,
    null_layer,
)
from rakshak.generator.engine import GeneratedData, generate
from rakshak.schemas import ConfounderId, Instrument, PersonaId

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
START = datetime(2026, 1, 1, tzinfo=UTC)
SEED = 13
N_MERCHANTS = 1_500
#: The Done-when threshold: population mean |z| inside every event window.
Z_FLOOR = 1.0


@pytest.fixture(scope="module")
def config() -> ScenarioConfig:
    base = load_scenario(CONFIG_PATH)
    return dataclasses.replace(
        base,
        population=dataclasses.replace(
            base.population, n_merchants=N_MERCHANTS, prevalence=0.0
        ),
    )


@pytest.fixture(scope="module")
def data(config: ScenarioConfig) -> GeneratedData:
    return generate(config, np.random.default_rng(SEED))


@pytest.fixture(scope="module")
def windows(config: ScenarioConfig) -> list[ConfounderWindow]:
    """The schedule depends only on the config, so it can be built without a run — which
    is the point of ``build_layer`` taking no ``rng``: the gate reads the same schedule
    the generator did, out of the same file, rather than being told it afterwards."""
    persona_idx = np.zeros(4, dtype=np.int64)
    layer = build_layer(config, persona_idx, np.full(4, 6.0), np.full(4, 0.5))
    return layer.windows


# ─────────────────────────────────────────────────────────────────────────────
# 1. Independence — structural, not conventional
# ─────────────────────────────────────────────────────────────────────────────


def imported_modules(module: object) -> set[str]:
    source = Path(module.__file__).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_persona_and_typology_code_cannot_see_the_confounder_layer() -> None:
    """The structural half of the ``prevalence=0, confounders=on`` null test.

    If a confounder effect were computed anywhere inside the persona code, "the
    confounder layer is independent" would be documentation. This makes it a fact about
    the import graph, checked by the AST rather than by reading.
    """
    for module in (personas_mod, typologies_mod):
        assert not any(
            "confounder" in name for name in imported_modules(module)
        ), f"{module.__name__} imports the confounder layer"


def test_the_confounder_layer_cannot_see_who_is_fraudulent() -> None:
    """And the other direction: a confounder that knew which merchants had turned could
    make the null test pass by simply not firing on them."""
    names = [n for n in imported_modules(confounders_mod) if "schemas" not in n]
    assert not any("typolog" in n or "persona" in n.lower() for n in names)


def test_layer_is_disabled_by_configuration_not_by_code(config: ScenarioConfig) -> None:
    """The rollback clause of the ticket: ``confounders.enabled = false`` and the rest of
    the generator is unaffected."""
    off = dataclasses.replace(
        config, confounders=dataclasses.replace(config.confounders, enabled=False)
    )
    layer = build_layer(off, np.zeros(5, dtype=np.int64), np.full(5, 6.0), np.full(5, 0.5))
    assert layer.windows == []
    np.testing.assert_array_equal(layer.intensity, np.ones((5, config.population.n_days)))
    assert not layer.enabled
    null = null_layer(5, config.population.n_days)
    np.testing.assert_array_equal(layer.intensity, null.intensity)


def test_all_six_confounders_are_scheduled(windows: list[ConfounderWindow]) -> None:
    assert {w.confounder for w in windows} == set(ConfounderId)
    for window in windows:
        assert window.end_day > window.start_day


# ─────────────────────────────────────────────────────────────────────────────
# 2. Effect size — mean |z| > 1.0 inside every window
# ─────────────────────────────────────────────────────────────────────────────


def daily_feature(
    data: GeneratedData, config: ScenarioConfig, feature: str
) -> np.ndarray:
    """``(n_merchants, n_days)`` matrix of the observable a confounder window names.

    Share-valued features are NaN on a merchant-day with no transactions — a share of
    nothing is not zero, and treating it as zero would manufacture a shift every time a
    confounder changed the *volume*.
    """
    n, n_days = config.population.n_merchants, config.population.n_days
    frame = data.transactions.with_columns(
        m=pl.col("merchant_id").str.slice(1).cast(pl.Int64),
        day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int64),
    ).filter(~pl.col("is_refund"))

    if feature == "txn_count":
        agg = frame.group_by(["m", "day"]).len().rename({"len": "value"})
        out = np.zeros((n, n_days))
        out[agg["m"].to_numpy(), agg["day"].to_numpy()] = agg["value"].to_numpy()
        return out

    numerator = {
        "auth_fail_rate": (pl.col("status") == "failed"),
        "cnp_share": pl.col("is_cnp"),
        "instrument_mix": (
            pl.col("instrument") == config.confounders.P3_fee_change.target_instrument
        ),
        "new_instrument_share": (
            pl.col("instrument") == config.confounders.P4_new_method.target_instrument
        ),
    }[feature]
    agg = frame.group_by(["m", "day"]).agg(numerator.mean().alias("value"))
    out = np.full((n, n_days), np.nan)
    out[agg["m"].to_numpy(), agg["day"].to_numpy()] = agg["value"].to_numpy()
    return out


def window_z(matrix: np.ndarray, window: ConfounderWindow, step_like: bool) -> np.ndarray:
    """Per-merchant z of the window mean against that merchant's own baseline.

    The window mean, and the baseline sd divided by sqrt(window length), because the
    confounder magnitudes are stated in sigma units of the feature *read over the event's
    own duration* — a five-day festival is a five-day observation, not five one-day ones.

    ``step_like`` confounders (P3, P4, P5) are permanent, so their baseline is the days
    *before* the step; using all non-window days would put the post-step level into the
    baseline and shrink the very effect being measured.
    """
    n_days = matrix.shape[1]
    days = np.arange(n_days)
    if step_like:
        baseline_days = days < window.start_day
    else:
        baseline_days = (days < window.start_day) | (days >= window.end_day)

    baseline = matrix[:, baseline_days]
    window_slice = matrix[:, window.start_day : window.end_day]
    with warnings.catch_warnings():
        # A merchant with no transactions inside the window has no share to average.
        warnings.simplefilter("ignore", RuntimeWarning)
        base_mean = np.nanmean(baseline, axis=1)
        base_sd = np.nanstd(baseline, axis=1, ddof=1)
        observed = np.nanmean(window_slice, axis=1)
    width = window.end_day - window.start_day
    # Merchants with a degenerate baseline are dropped, not floored. L4's instrument mix
    # has wallet at exactly 0.00, so its pre-launch wallet share has zero variance and a
    # z against it is arithmetically infinite and statistically meaningless — flooring the
    # denominator instead reported P4 at |z| = 1.6e7, which is a bug wearing a strong
    # result's clothes.
    usable = np.isfinite(base_sd) & (base_sd > 0.0) & np.isfinite(observed) & np.isfinite(base_mean)
    z = (observed[usable] - base_mean[usable]) / (base_sd[usable] / np.sqrt(width))
    return z


STEP_LIKE = {ConfounderId.P3, ConfounderId.P4, ConfounderId.P5}


def test_every_confounder_window_moves_the_population(
    data: GeneratedData, config: ScenarioConfig, windows: list[ConfounderWindow]
) -> None:
    """**The Done-when clause.** Population mean |z| > 1.0 inside every event window, at
    prevalence 0 — so every bit of that movement is platform, not fraud."""
    measured: dict[str, float] = {}
    for window in windows:
        matrix = daily_feature(data, config, window.feature)
        z = window_z(matrix, window, window.confounder in STEP_LIKE)
        key = f"{window.confounder.value}@{window.start_day}-{window.end_day}({window.feature})"
        measured[key] = float(np.mean(np.abs(z)))

    weak = {k: round(v, 3) for k, v in measured.items() if v <= Z_FLOOR}
    assert not weak, (
        f"confounder windows below the |z| > {Z_FLOOR} floor: {weak}. All measured: "
        f"{ {k: round(v, 3) for k, v in measured.items()} }"
    )


def test_outside_every_window_the_population_is_quiet(
    data: GeneratedData, config: ScenarioConfig, windows: list[ConfounderWindow]
) -> None:
    """The control. If |z| were above the floor on an arbitrary quiet stretch too, the
    test above would be measuring the noise rather than the confounder."""
    matrix = daily_feature(data, config, "txn_count")
    busy = np.zeros(config.population.n_days, dtype=bool)
    for window in windows:
        busy[max(0, window.start_day - 3) : window.end_day + 3] = True
    quiet = np.flatnonzero(~busy)
    assert quiet.size > 20, "no quiet stretch left to use as a control"
    placebo = ConfounderWindow(ConfounderId.P1, int(quiet[5]), int(quiet[5]) + 5, "txn_count")
    z = window_z(matrix, placebo, step_like=False)
    assert float(np.mean(np.abs(z))) < Z_FLOOR


# ─────────────────────────────────────────────────────────────────────────────
# 3. Heterogeneity — persona_sensitivity varies the effect size
# ─────────────────────────────────────────────────────────────────────────────


def test_persona_sensitivity_varies_the_effect_across_personas(
    data: GeneratedData, config: ScenarioConfig, windows: list[ConfounderWindow]
) -> None:
    """P1 must hit L2 (seasonal D2C, sensitivity 1.70) harder than L4 (lumpy B2B, 0.45).

    Without this the cohort residual has an unfairly easy job: if every merchant moved by
    exactly the same amount, subtracting the cohort median would erase the confounder
    perfectly and G5 would pass for a reason that will not survive contact with a real
    portfolio.
    """
    festival = next(w for w in windows if w.confounder is ConfounderId.P1)
    matrix = daily_feature(data, config, "txn_count")
    z = np.full(matrix.shape[0], np.nan)
    finite = window_z(matrix, festival, step_like=False)
    # window_z drops non-finite rows; recompute aligned to keep the persona join simple.
    n_days = matrix.shape[1]
    days = np.arange(n_days)
    base = matrix[:, (days < festival.start_day) | (days >= festival.end_day)]
    width = festival.end_day - festival.start_day
    observed = matrix[:, festival.start_day : festival.end_day].mean(axis=1)
    z = (observed - base.mean(axis=1)) / np.maximum(
        base.std(axis=1, ddof=1) / np.sqrt(width), 1e-9
    )
    assert finite.size > 0

    persona = data.ground_truth["persona_id"].to_numpy()
    means = {
        pid: float(np.nanmean(z[persona == pid.value]))
        for pid in (PersonaId.L2, PersonaId.L4, PersonaId.L5)
    }
    assert means[PersonaId.L2] > means[PersonaId.L4], means
    # And subscriptions do not care that it is Diwali.
    assert means[PersonaId.L2] > means[PersonaId.L5], means


def test_confounders_reach_every_merchant_not_just_a_sample(
    data: GeneratedData, config: ScenarioConfig, windows: list[ConfounderWindow]
) -> None:
    """"Platform-wide" means platform-wide. A confounder that fires on a random subset is
    a second typology, and the cohort residual would be able to average it away."""
    festival = next(w for w in windows if w.confounder is ConfounderId.P1)
    matrix = daily_feature(data, config, "txn_count")
    z = window_z(matrix, festival, step_like=False)
    # Three quarters, not all of them, and the shortfall is the arrival process rather
    # than a leaky layer: at Fano 12.25 a merchant's own five-day noise is comparable to
    # a 3-sigma platform event, so roughly a fifth of the population still posts a
    # negative z during a real festival. That is what makes the confounder hard to tell
    # from fraud, which is the entire premise of gate G5.
    assert float((z > 0).mean()) > 0.75, f"only {(z > 0).mean():.1%} of merchants moved up"
    assert float(np.median(z)) > 0.5


def test_p3_and_p4_move_the_instrument_mix_in_the_right_direction(
    data: GeneratedData, config: ScenarioConfig
) -> None:
    """P3 shifts share toward the cheaper rail and P4 toward a newly launched one. The
    directions are asserted because a sign error here would still produce a large |z| and
    would still pass the test above."""
    frame = data.transactions.with_columns(
        day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int64)
    )
    p3, p4 = config.confounders.P3_fee_change, config.confounders.P4_new_method

    def share(instrument: str, lo: int, hi: int) -> float:
        rows = frame.filter((pl.col("day") >= lo) & (pl.col("day") < hi))
        return float((rows["instrument"] == instrument).mean())

    assert share(p3.target_instrument, p3.day, p3.day + p3.window_days) > share(
        p3.target_instrument, p3.day - p3.window_days, p3.day
    )
    after = share(p4.target_instrument, p4.day + p4.ramp_days, p4.day + 2 * p4.ramp_days)
    assert after > share(p4.target_instrument, p4.day - p4.ramp_days, p4.day)
    assert p3.target_instrument in {i.value for i in Instrument}
    assert p4.target_instrument in {i.value for i in Instrument}
