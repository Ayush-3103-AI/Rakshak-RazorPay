"""T-115: delayed, noisy and censored labels, and the invariant that makes v2 honest.

The hard invariant is::

    label_available_at > label_event_at >= drift_onset_at

The first leg is enforced in ``schemas.Label.__post_init__``. The second **cannot** be:
``drift_onset_at`` is quarantined in ``GroundTruth`` and must not be reachable from
``Label``, so it can only be checked across the join — which is exactly what this file
does, and why it does it with hypothesis rather than three examples. An off-by-one on a
``>=`` is the bug an example-based test picks the wrong example for.

v1 trained against labels the instant the fraud occurred. That measured a system that
cannot exist, and it is the single change here that most directly changes what the
downstream numbers mean.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rakshak.generator.config import LabelsConfig, load_scenario
from rakshak.generator.engine import generate
from rakshak.generator.labels import NO_TIME, emit_labels
from rakshak.schemas import Label, LabelSource

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
START = datetime(2026, 1, 1, tzinfo=UTC)
NS_PER_DAY = 86_400_000_000_000
SIM_START = int(START.timestamp()) * 1_000_000_000
N_DAYS = 180
SIM_END = SIM_START + N_DAYS * NS_PER_DAY


@pytest.fixture(scope="module")
def labels_config() -> LabelsConfig:
    return load_scenario(CONFIG_PATH).labels


def onsets(rng: np.random.Generator, n: int, fraud_rate: float) -> np.ndarray:
    """A drift-onset array with ``NO_TIME`` for the merchants that never turned."""
    is_fraud = rng.random(n) < fraud_rate
    days = rng.integers(0, N_DAYS, size=n)
    return np.where(is_fraud, SIM_START + days * NS_PER_DAY, NO_TIME)


# ─────────────────────────────────────────────────────────────────────────────
# The invariant, property-tested
# ─────────────────────────────────────────────────────────────────────────────


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n=st.integers(min_value=50, max_value=800),
    fraud_rate=st.floats(min_value=0.0, max_value=1.0),
    unreported=st.floats(min_value=0.0, max_value=1.0),
    spurious=st.floats(min_value=0.0, max_value=0.5),
    dispute_mean=st.floats(min_value=0.5, max_value=90.0),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_label_time_invariant_holds_everywhere(
    labels_config: LabelsConfig,
    seed: int,
    n: int,
    fraud_rate: float,
    unreported: float,
    spurious: float,
    dispute_mean: float,
) -> None:
    """``label_available_at > label_event_at >= drift_onset_at`` on every row, across the
    whole parameter surface — including the degenerate corners (all fraud, none reported,
    a half-day dispute delay) where an ordering bug would actually live."""
    rng = np.random.default_rng(seed)
    cfg = dataclasses.replace(
        labels_config,
        unreported_rate=unreported,
        spurious_chargeback_rate=spurious,
        fraud_to_dispute_mean_days=dispute_mean,
    )
    drift = onsets(rng, n, fraud_rate)
    draw = emit_labels(rng, cfg, drift_onset_ns=drift, sim_start_ns=SIM_START, sim_end_ns=SIM_END)

    dated = draw.label_event_ns != NO_TIME
    assert np.all(draw.label_available_ns[dated] > draw.label_event_ns[dated])
    # The leg that cannot live in schemas.Label, checked across the join.
    both = dated & (drift != NO_TIME)
    assert np.all(draw.label_event_ns[both] >= drift[both])
    # A row is dated iff it is available-dated: there is no half-populated label.
    assert np.array_equal(dated, draw.label_available_ns != NO_TIME)


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n=st.integers(min_value=50, max_value=400),
    fraud_rate=st.floats(min_value=0.0, max_value=1.0),
)
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_every_row_constructs_as_a_schema_label(
    labels_config: LabelsConfig, seed: int, n: int, fraud_rate: float
) -> None:
    """Every emitted row must survive ``schemas.Label.__post_init__``.

    Cheaper than restating those invariants here, and strictly stronger: the contract
    cannot drift away from the generator without this failing.
    """
    rng = np.random.default_rng(seed)
    drift = onsets(rng, n, fraud_rate)
    draw = emit_labels(
        rng, labels_config, drift_onset_ns=drift, sim_start_ns=SIM_START, sim_end_ns=SIM_END
    )
    for i in range(0, n, max(1, n // 40)):
        Label(
            merchant_id=f"M{i:06d}",
            label=None if np.isnan(draw.label[i]) else int(draw.label[i]),
            label_event_at=None
            if draw.label_event_ns[i] == NO_TIME
            else datetime.fromtimestamp(draw.label_event_ns[i] / 1e9, tz=UTC),
            label_available_at=None
            if draw.label_available_ns[i] == NO_TIME
            else datetime.fromtimestamp(draw.label_available_ns[i] / 1e9, tz=UTC),
            label_source=LabelSource(draw.source[i]),
            is_censored=bool(draw.is_censored[i]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# All four states, and the two noise rates
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def population(labels_config: LabelsConfig) -> tuple[np.ndarray, object]:
    """10,000 merchants at the real prevalence — the sample size the ticket names for
    the rate assertions."""
    rng = np.random.default_rng(97)
    n = 10_000
    # Onsets early enough that a dispute can resolve inside the horizon, so the observed
    # and censored states are both populated rather than one of them being empty.
    is_fraud = rng.random(n) < 0.25
    drift = np.where(is_fraud, SIM_START + rng.integers(0, 60, size=n) * NS_PER_DAY, NO_TIME)
    return drift, emit_labels(
        rng, labels_config, drift_onset_ns=drift, sim_start_ns=SIM_START, sim_end_ns=SIM_END
    )


def test_all_four_label_states_are_produced(population: tuple) -> None:
    """08-generator-v2-spec.md §6 requires all four. Two of them — the unreported
    positive and the spurious one — are what make this weak supervision rather than a
    clean classification task, and a run that silently produced neither would look
    entirely healthy."""
    drift, draw = population
    is_fraud = drift != NO_TIME

    observed_positive = int(((draw.label == 1.0) & ~draw.is_censored & is_fraud).sum())
    unreported_positive = int(draw.is_unreported.sum())
    censored = int(draw.is_censored.sum())
    spurious = int(((draw.label == 1.0) & ~is_fraud).sum())

    assert observed_positive > 0
    assert unreported_positive > 0
    assert censored > 0
    assert spurious > 0
    # An unreported positive is labelled 0 and is NOT censored: it is a *wrong* label,
    # not a missing one, and conflating the two would delete the noise the harness exists
    # to expose.
    assert np.all(draw.label[draw.is_unreported] == 0.0)
    assert not np.any(draw.is_censored & draw.is_unreported)
    # A censored row carries no resolved label.
    assert np.all(np.isnan(draw.label[draw.is_censored]))


def test_unreported_rate_is_honoured(
    population: tuple, labels_config: LabelsConfig
) -> None:
    drift, draw = population
    is_fraud = drift != NO_TIME
    rate = float(draw.is_unreported.sum() / is_fraud.sum())
    n = int(is_fraud.sum())
    target = labels_config.unreported_rate
    tolerance = 4.0 * np.sqrt(target * (1 - target) / n)
    assert abs(rate - target) < tolerance, f"unreported rate {rate:.4f} vs {target}"


def test_spurious_chargeback_rate_is_honoured(
    population: tuple, labels_config: LabelsConfig
) -> None:
    """0.3% of good merchants get a chargeback anyway. A model that assumes its labels
    are correct will overfit to exactly this, and the harness should be able to show it."""
    drift, draw = population
    legit = drift == NO_TIME
    rate = float(((draw.label[legit] == 1.0) | draw.is_censored[legit]).sum() / legit.sum())
    n = int(legit.sum())
    target = labels_config.spurious_chargeback_rate
    tolerance = 4.0 * np.sqrt(target * (1 - target) / n)
    assert abs(rate - target) < tolerance, f"spurious rate {rate:.5f} vs {target}"


def test_labels_are_never_available_the_instant_they_occur(population: tuple) -> None:
    """The delay is not decorative. The narrowest gap in the population must still be at
    least the configured minimum dispute delay."""
    _, draw = population
    dated = draw.label_event_ns != NO_TIME
    gap_days = (draw.label_available_ns[dated] - draw.label_event_ns[dated]) / NS_PER_DAY
    assert gap_days.min() >= 45.0
    assert gap_days.max() <= 120.0


def test_no_label_is_available_after_the_simulation_ends(population: tuple) -> None:
    """Anything resolving past the horizon is censored, by definition. A row that is both
    available-in-the-future and labelled would be a label from beyond the data."""
    _, draw = population
    dated = draw.label_available_ns != NO_TIME
    late = dated & (draw.label_available_ns > SIM_END)
    assert np.all(draw.is_censored[late])
    assert np.all(np.isnan(draw.label[late]))


# ─────────────────────────────────────────────────────────────────────────────
# End to end, through the engine
# ─────────────────────────────────────────────────────────────────────────────


def test_invariant_survives_the_full_generator() -> None:
    """The same invariant, but across the real join of the two persisted tables — which
    is where a timezone or dtype mistake would surface rather than in the numpy arrays."""
    config = load_scenario(CONFIG_PATH)
    config = dataclasses.replace(
        config,
        population=dataclasses.replace(config.population, n_merchants=800, prevalence=0.3),
        confounders=dataclasses.replace(config.confounders, enabled=False),
    )
    data = generate(config, np.random.default_rng(5))
    joined = data.labels.join(data.ground_truth, on="merchant_id")
    assert joined.height == 800
    assert (
        joined.filter(
            pl.col("label_available_at").is_not_null()
            & (pl.col("label_available_at") <= pl.col("label_event_at"))
        ).height
        == 0
    )
    assert (
        joined.filter(
            pl.col("label_event_at").is_not_null()
            & pl.col("drift_onset_at").is_not_null()
            & (pl.col("label_event_at") < pl.col("drift_onset_at"))
        ).height
        == 0
    )
    for column in ("label_event_at", "label_available_at"):
        assert data.labels.schema[column].time_zone == "UTC"
        assert data.labels.schema[column].time_unit == "ns"
