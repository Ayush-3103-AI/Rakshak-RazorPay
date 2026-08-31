"""T-121: the residual layer agrees whichever runner produced the z it was given.

The residual itself has no online and offline form — it is one cross-merchant pass per
epoch and there is only one of it. What has to hold is that **the layer is transparent to
the runner**: residualising the online z-vector and residualising the offline z-vector give
the same answer, for every flagged feature, on every epoch.

That is a weaker statement than the T1 parity suite and it is the right one. If it ever
failed while `test_tier1_parity.py` was green, the fault would be in the residual layer
depending on something other than the values it is handed — a dict iteration order, a NaN
policy, a cohort built from something that moves. Which is exactly the class of bug that
would be invisible until a model trained offline scored differently online.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from conftest import end_of_day, epochs_between, synthetic_stream, to_frame
from rakshak.features import cohort, registry, tier1
from rakshak.features.spec import PARITY_TOLERANCE
from rakshak.features.state import MerchantState
from rakshak.schemas import MerchantProfile, Transaction

#: `synthetic_stream` gives six merchants in one `mcc_group`, so the real 30-member chain
#: would put every one of them in `global` and the test would only ever exercise one cohort.
#: Lowering the floor is what makes the full-key and backoff legs reachable at this scale;
#: the production value is asserted separately in tests/unit/test_cohort.py.
TEST_MIN_MEMBERS = 2
TEST_WARMUP = 5

FLAGGED = cohort.residual_features()


@pytest.fixture
def loaded(rng: np.random.Generator) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    txns, profiles = synthetic_stream(rng, days=40)
    tier1.load_profiles(profiles)
    return txns, profiles


def test_there_are_features_flagged_for_a_residual() -> None:
    assert FLAGGED, "no T1 feature carries has_cohort_residual — the layer has no inputs"
    print(f"\n{len(FLAGGED)} flagged features: {', '.join(FLAGGED)}")


def test_residuals_agree_whichever_runner_produced_the_z(
    loaded: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """Replay every flagged feature both ways and residualise both, epoch by epoch."""
    txns, profiles = loaded
    frame = to_frame(txns)
    days = epochs_between(min(t.event_date for t in txns), max(t.event_date for t in txns))
    asg = cohort.assign_cohorts(profiles, min_members=TEST_MIN_MEMBERS)

    specs = [type(registry.get(name))(warmup_days=TEST_WARMUP) for name in FLAGGED]
    states = {mid: MerchantState(merchant_id=mid, profile=p) for mid, p in profiles.items()}
    ordered = sorted(txns, key=lambda t: (t.event_time, t.event_id))

    worst = 0.0
    cursor = 0
    for day in days:
        as_of = end_of_day(day)
        while cursor < len(ordered) and ordered[cursor].event_time <= as_of:
            event = ordered[cursor]
            for spec in specs:
                spec.update(spec.state_of(states[event.merchant_id]), event)
            cursor += 1

        prefix = frame.filter(pl.col("event_time") <= as_of).lazy()
        for spec in specs:
            online = {
                mid: spec.value(spec.state_of(state), as_of) for mid, state in states.items()
            }
            out = spec.batch(prefix, as_of)
            lookup = dict(
                zip(out["merchant_id"].to_list(), out[spec.name].to_list(), strict=True)
            )
            offline = {mid: (lookup.get(mid) or 0.0) for mid in states}

            r_on = cohort.residuals(asg, online)
            r_off = cohort.residuals(asg, offline)
            assert set(r_on) == set(r_off)
            for mid in r_on:
                worst = max(worst, abs(r_on[mid] - r_off[mid]))

    print(f"\nmax residual parity diff across {len(FLAGGED)} flagged features: {worst:.3e}")
    assert worst <= PARITY_TOLERANCE


def test_the_residual_block_is_column_aligned_with_its_base_features(
    loaded: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """`residual_matrix` must not reorder anything.

    The residual columns are appended to the feature vector in the base features' order
    (09-interfaces.md §9). A layer that sorted its cohorts, or its merchants, into a
    different order than it was handed would produce a feature vector that is correct
    row-wise and scrambled column-wise — and would train and score without complaining.
    """
    _, profiles = loaded
    asg = cohort.assign_cohorts(profiles, min_members=TEST_MIN_MEMBERS)
    ids = sorted(profiles)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(len(ids), len(FLAGGED)))
    matrix = cohort.residual_matrix(asg, ids, z)
    assert matrix.shape == z.shape
    for col, name in enumerate(FLAGGED):
        single = cohort.residuals(asg, {m: float(z[i, col]) for i, m in enumerate(ids)})
        assert np.allclose(matrix[:, col], [single[m] for m in ids], atol=1e-12), name


def test_the_silent_merchant_does_not_drag_its_cohort_median(
    loaded: tuple[list[Transaction], dict[str, MerchantProfile]],
) -> None:
    """The stream's last merchant never transacts, and its z is 0.0 on every epoch.

    That 0.0 is a real value, not a missing one — the online contract says a merchant with
    nothing seen reads 0.0 — so it is a legitimate cohort member and it must be counted.
    The distinction that matters is between "read 0.0" and "was not read", and only the
    second is dropped. This test pins which one the layer is doing.
    """
    _, profiles = loaded
    asg = cohort.assign_cohorts(profiles, min_members=TEST_MIN_MEMBERS)
    silent = max(profiles)
    values = {m: 4.0 for m in profiles}
    values[silent] = 0.0
    r_with = cohort.residuals(asg, values)
    r_without = cohort.residuals(asg, {m: v for m, v in values.items() if m != silent})
    assert silent in r_with
    assert silent not in r_without
    assert r_with[silent] == pytest.approx(-4.0)
