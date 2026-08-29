"""BAF validation of the decision layer (T-0012, FR-021).

Seams under test, confirmed with the user before writing:

1. `baf_cost_params` - the mapping from BAF's one monetary column onto the cost
   layer's `L` and `V`. Hand-computed on a 3-row fixture; every expected number
   below is a literal with its arithmetic in the comment beside it.
2. `split_baf` - BAF's **native** temporal split. The ticket is explicit that the
   synthetic split from T-0005 must not be used here, so the guard is that the
   months partition, are disjoint, and that `test` is month 7 alone.

The scoring run itself is not tested here; it is a composition of `models.gbdt`
and `decision.policy`, both already pinned by their own suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak import config
from rakshak.eval.baf import baf_cost_params, split_baf

# proposed_credit_limit is BAF's only monetary column. Three rows, three limits.
_FIXTURE = pd.DataFrame(
    {
        "fraud_bool": [0, 1, 0],
        "proposed_credit_limit": [200.0, 1000.0, 2000.0],
        "month": [0, 6, 7],
        "credit_risk_score": [100, 200, 300],
    }
)


def test_loss_is_exposure_times_realisation_and_loading():
    """L_i = credit_limit_i * r_cb * (1 + phi). Hand-computed, not recomputed."""
    params = baf_cost_params(_FIXTURE)
    r_cb = config.CHARGEBACK_REALISATION_RATE  # 0.05
    phi = config.ANCILLARY_LOADING_PHI
    # 200 * 0.05 * (1 + phi); 1000 * 0.05 * (1 + phi); 2000 * 0.05 * (1 + phi)
    expected = np.array([200.0, 1000.0, 2000.0]) * r_cb * (1.0 + phi)
    np.testing.assert_allclose(params.loss_inr, expected, rtol=1e-12)


def test_value_is_exposure_times_margin_times_lifetime():
    """V_i = credit_limit_i * g * lifetime_months. Hand-computed."""
    params = baf_cost_params(_FIXTURE)
    g = config.GROSS_MARGIN_RATE
    lifetime = config.MERCHANT_LIFETIME_MONTHS
    expected = np.array([200.0, 1000.0, 2000.0]) * g * lifetime
    np.testing.assert_allclose(params.value_inr, expected, rtol=1e-12)


def test_example_dependence_survives_only_through_the_flat_support_cost():
    """L and V are both linear in the same column, so the ONLY thing that makes
    per-application thresholds differ is the flat `c_support` term in `c_fp`.

    This is a real weakness of the BAF mapping relative to the synthetic layer,
    where volume and lifetime vary independently. It is asserted here so that if
    someone later makes V depend on another column, this test fails and the
    results file's caveat gets revisited rather than silently going stale.
    """
    from rakshak.decision.policy import hold_threshold

    params = baf_cost_params(_FIXTURE)
    thresholds = hold_threshold(params)
    assert len({round(float(t), 12) for t in thresholds}) > 1, (
        "thresholds are identical across applications - the flat support cost is "
        "no longer breaking proportionality, so the results file's caveat is wrong"
    )

    # And prove the claim about *why*: with c_support zeroed, they collapse to one.
    from dataclasses import replace

    flat = hold_threshold(replace(params, cost_support_inr=0.0))
    assert len({round(float(t), 12) for t in flat}) == 1


def test_split_is_bafs_native_temporal_split_not_the_synthetic_one():
    """months 0-5 train, 6 validate, 7 test. Disjoint, and they partition the frame."""
    frame = pd.DataFrame({"month": list(range(8)), "fraud_bool": [0] * 8})
    splits = split_baf(frame)

    assert sorted(splits.train["month"].unique()) == [0, 1, 2, 3, 4, 5]
    assert sorted(splits.validate["month"].unique()) == [6]
    assert sorted(splits.test["month"].unique()) == [7]

    total = len(splits.train) + len(splits.validate) + len(splits.test)
    assert total == len(frame), "splits must partition the frame, losing no row"


def test_split_rejects_a_frame_missing_the_month_column():
    """BAF's temporal provenance is the whole point; failing loudly beats guessing."""
    with pytest.raises(KeyError):
        split_baf(pd.DataFrame({"fraud_bool": [0, 1]}))
