"""Rung 8 — the realised-exposure decision-policy wrapper.

Pre-registered in ``docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`` §4.2. The tests here
assert externally observable properties of a scored result, not that a function was called:
that the capacity budget K survives an input substitution, that the reconstruction of
trailing GMV is exact rather than approximate, and that the wrapper is a provable no-op when
handed the exposure the inner policy would have used anyway.

That last one is the important one. It is the golden-output check that makes the A/B in the
pre-registration a controlled comparison: arm A and arm B must be *identical* when the two
exposure vectors agree, or any difference between the arms could be the wrapper rather than
the exposure.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.eval.capacity import (
    DEFAULT_DECISION,
    CostParams,
    DecisionRequest,
    select_actions,
)
from rakshak.models.decision_realised_exposure import (
    RealisedExposure,
    realised_exposure_inr,
)
from rakshak.schemas import Action

PARAMS = CostParams()


def _request(n: int, days: int, k: int, rng: np.random.Generator) -> DecisionRequest:
    return DecisionRequest(
        score=rng.random(n),
        day=np.repeat(np.arange(days), n // days),
        # Heavy-tailed, like the real GMV distribution: a uniform exposure would make the
        # exposure term irrelevant and the test vacuous.
        exposure_inr=np.exp(rng.normal(10.0, 1.5, size=n)),
        k=k,
        params=PARAMS,
    )


def test_reconstruction_of_trailing_gmv_is_exact() -> None:
    """``v_declared_ratio`` is ``trailing-30d GMV / declared_monthly_gmv`` by definition, so
    the product must return the numerator to floating-point equality — not approximately."""
    rng = np.random.default_rng(3)
    declared = np.exp(rng.normal(10.0, 1.5, size=500))
    trailing = np.exp(rng.normal(10.0, 1.5, size=500))
    ratio = trailing / declared
    np.testing.assert_allclose(
        realised_exposure_inr(declared, ratio), trailing, rtol=1e-12, atol=0.0
    )


def test_a_merchant_with_no_declaration_falls_back_to_the_incumbent_estimator() -> None:
    """``DeclaredRatio.read`` returns 0.0 when ``declared_gmv <= 0``. Reconstructing from
    that would price the merchant at zero exposure, which makes it unalertable at any score.
    The fallback must be the declared figure — cycle 3's estimator — so the wrapper is never
    worse than the incumbent on those rows."""
    declared = np.array([0.0, 100.0, 250_000.0])
    ratio = np.array([0.0, 0.0, 2.0])  # first two are the degenerate cases
    out = realised_exposure_inr(declared, ratio)
    assert out[0] == 0.0  # nothing declared and nothing trailing: genuinely zero
    assert out[1] == 100.0  # declared but no trailing GMV yet -> fall back, do not zero it
    assert out[2] == 500_000.0  # the normal case


def test_capacity_k_survives_the_substitution() -> None:
    """The seam's binding rule. This wrapper does not soften actions — it substitutes an
    input and lets the inner policy select — so K is preserved by construction. Asserted at
    the boundary rather than argued in a docstring."""
    rng = np.random.default_rng(11)
    k = 15
    req = _request(n=6000, days=60, k=k, rng=rng)
    exposure = np.exp(rng.normal(10.0, 2.0, size=6000))
    actions = RealisedExposure(inner=DEFAULT_DECISION, exposure=exposure).decide(req)
    for d in np.unique(req.day):
        non_pass = int(np.sum(actions[req.day == d] != Action.PASS))
        assert non_pass <= k, f"day {d} emitted {non_pass} non-PASS actions against K={k}"


def test_the_wrapper_is_a_no_op_when_the_exposures_agree() -> None:
    """Arm A and arm B must differ *only* by the exposure vector.

    If this fails, every difference the pre-registered A/B measures is confounded with the
    wrapper itself, and the comparison says nothing about exposure.
    """
    rng = np.random.default_rng(7)
    req = _request(n=4000, days=40, k=12, rng=rng)
    wrapped = RealisedExposure(inner=DEFAULT_DECISION, exposure=req.exposure_inr).decide(req)
    direct = select_actions(
        req.score, req.day, req.exposure_inr, req.k, req.params, req.hold_policy
    )
    np.testing.assert_array_equal(wrapped, direct)


def test_a_better_exposure_estimate_captures_more_loss_at_the_same_k() -> None:
    """The mechanism §8.3a claims, as a property rather than as a citation of a measurement.

    Construct merchants whose realised loss scales with a *true* exposure. Give one policy a
    noisy proxy for it (the declaration) and the other the true value, hold the score and K
    identical, and the policy with the better exposure estimate must avert more rupees. If
    this does not hold, the reasoning behind Rung 8 is wrong and the rung should not exist.
    """
    rng = np.random.default_rng(5)
    n, days, k = 4000, 40, 10
    true_exposure = np.exp(rng.normal(10.0, 1.5, size=n))
    # The generator's own corruption: lognormal, sigma 0.55.
    declared = true_exposure * np.exp(rng.normal(0.0, 0.55, size=n))
    score = rng.random(n)
    day = np.repeat(np.arange(days), n // days)
    base = DecisionRequest(
        score=score, day=day, exposure_inr=declared, k=k, params=PARAMS
    )

    def loss_averted(exposure: np.ndarray) -> float:
        actions = RealisedExposure(inner=DEFAULT_DECISION, exposure=exposure).decide(base)
        return float(np.sum(np.where(actions != Action.PASS, score * true_exposure, 0.0)))

    assert loss_averted(true_exposure) > loss_averted(declared)


def test_a_misaligned_exposure_vector_is_refused_rather_than_broadcast() -> None:
    rng = np.random.default_rng(2)
    req = _request(n=1000, days=10, k=5, rng=rng)
    policy = RealisedExposure(inner=DEFAULT_DECISION, exposure=np.ones(999))
    with pytest.raises(ValueError, match="rows but the request has"):
        policy.decide(req)


def test_a_negative_exposure_is_refused() -> None:
    rng = np.random.default_rng(2)
    req = _request(n=100, days=10, k=5, rng=rng)
    bad = np.ones(100)
    bad[7] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        RealisedExposure(inner=DEFAULT_DECISION, exposure=bad).decide(req)


def test_the_policy_names_itself_in_terms_of_what_it_wraps() -> None:
    """A result produced under a wrapped policy must never be mistakable for one produced
    under the default — the name is what appears beside the number."""
    assert (
        RealisedExposure(inner=DEFAULT_DECISION, exposure=np.ones(1)).name
        == "realised_exposure(capacity_topk)"
    )
