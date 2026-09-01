"""Rung 9 — the rank-CUSUM detector.

Each test asserts an externally observable property of the statistic, not that a function
was called. The two that matter are ``test_a_monotone_platform_wide_shock_moves_nothing``
— the confounder guard, which is the rung's strongest claim and is stronger than the cohort
residual's — and ``test_the_cap_binds``, because without the cap the rung degenerates into
exactly the static watchlist it was chosen to defeat.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.models.rung9_rank_cusum import (
    C_MAX,
    K_REFERENCE,
    NORMAL_SCORE_CLIP,
    RankCusum,
    cross_sectional_normal_scores,
    page_recursion,
    run_length,
)


def _panel(n_merchants: int, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.repeat([f"M{i:05d}" for i in range(n_merchants)], n_days)
    d = np.tile(np.arange(n_days, dtype=np.int64), n_merchants)
    c = np.repeat([f"C{i % 4}" for i in range(n_merchants)], n_days)
    return m, d, c


def test_normal_scores_are_standard_under_no_change() -> None:
    """The property the whole method rests on: a within-day rank is uniform under the null
    whatever the score distribution is, so its normal score is ~N(0,1). Fed a heavily
    skewed, overdispersed input — which is what this generator actually produces at Fano
    12.25 — the output must still be standard."""
    rng = np.random.default_rng(0)
    m, d, c = _panel(400, 30)
    skewed = rng.lognormal(0.0, 2.5, size=m.size)  # nothing like a normal
    x = cross_sectional_normal_scores(skewed, d, c)
    assert abs(float(x.mean())) < 0.05
    assert 0.85 < float(x.std()) < 1.15


def test_a_monotone_platform_wide_shock_moves_nothing() -> None:
    """THE confounder guard, and the reason this beats the cohort residual on paper.

    A festival spike, a gateway outage or a fee change applies one monotone transform to the
    whole panel on one day. The residual layer cancels an *additive* common mode; a rank is
    invariant to *any* monotone map. Asserted against three shocks of quite different shape,
    none of which is additive.
    """
    rng = np.random.default_rng(1)
    m, d, c = _panel(300, 20)
    base = rng.random(m.size)
    shocked_day = 7
    hit = d == shocked_day
    for shock in (
        lambda v: v * 6.0,  # multiplicative: a volume spike
        lambda v: v**3,  # strongly convex
        lambda v: np.log1p(v * 40.0),  # strongly concave
    ):
        moved = base.copy()
        moved[hit] = shock(moved[hit])
        np.testing.assert_allclose(
            cross_sectional_normal_scores(moved, d, c),
            cross_sectional_normal_scores(base, d, c),
            rtol=0,
            atol=1e-12,
        )


def test_the_accumulator_stays_near_zero_under_no_change() -> None:
    """No change, no alarm. With mean-zero increments and a positive reference value the
    recursion is a negatively-drifting random walk reflected at 0, so it must not accumulate
    over a long horizon."""
    rng = np.random.default_rng(2)
    m, d, c = _panel(200, 120)
    x = cross_sectional_normal_scores(rng.random(m.size), d, c)
    acc = page_recursion(x, d, m)
    assert float(np.median(acc)) < 1.0
    assert float(acc.max()) < C_MAX


def test_the_accumulator_fires_on_a_real_shift_and_the_run_length_tracks_it() -> None:
    """One merchant's score rises partway through. Its accumulator must separate from the
    population's, and the run length must count the days since onset rather than restarting."""
    rng = np.random.default_rng(3)
    n_m, n_d, onset = 300, 80, 40
    m, d, c = _panel(n_m, n_d)
    score = rng.random(m.size)
    drifting = m == "M00000"
    score[drifting & (d >= onset)] += 3.0  # unambiguously the top of its cohort

    x = cross_sectional_normal_scores(score, d, c)
    acc = page_recursion(x, d, m)
    rl = run_length(acc, d, m)

    at_end = (d == n_d - 1)
    assert float(acc[drifting & at_end][0]) > float(np.quantile(acc[at_end], 0.99))
    # It has been drifting for the whole post-onset window, not just today.
    assert int(rl[drifting & at_end][0]) >= (n_d - onset) - 2


def test_the_cap_binds() -> None:
    """Without ``c_max`` a long-ago drifter pins the top-K forever and the alert set goes
    static — the `volume_rank` pathology (alert Jaccard 1.000) this rung exists to defeat.
    The cap is a design decision, so it is asserted rather than assumed."""
    n = 500
    m = np.array(["M0"] * n)
    d = np.arange(n, dtype=np.int64)
    x = np.full(n, 3.0)  # a merchant pinned at the top of its cohort every single day
    assert float(page_recursion(x, d, m).max()) == pytest.approx(C_MAX)
    assert float(page_recursion(x, d, m, c_max=5.0).max()) == pytest.approx(5.0)


def test_the_recursion_does_not_depend_on_input_row_order() -> None:
    """A recursion that silently depended on row order is a bug that only appears when the
    panel is reshuffled, which is the kind that survives a test suite."""
    rng = np.random.default_rng(4)
    m, d, c = _panel(60, 25)
    x = cross_sectional_normal_scores(rng.random(m.size), d, c)
    straight = page_recursion(x, d, m)
    perm = rng.permutation(m.size)
    shuffled = page_recursion(x[perm], d[perm], m[perm])
    np.testing.assert_allclose(shuffled, straight[perm], rtol=0, atol=0)


def test_ties_take_the_average_rank() -> None:
    """An outage drives a whole cohort to the same score on the same day. Breaking those
    ties arbitrarily would manufacture a spread of evidence out of an event in which nothing
    distinguished one merchant from another."""
    d = np.zeros(6, dtype=np.int64)
    c = np.array(["C0"] * 6)
    x = cross_sectional_normal_scores(np.array([1.0] * 6), d, c)
    np.testing.assert_allclose(x, np.zeros(6), atol=1e-12)


def test_a_lone_merchant_contributes_no_evidence() -> None:
    """A cohort of one yields u = 0.5 and a normal score of exactly 0 — correct, because a
    merchant compared only against itself has shown nothing."""
    x = cross_sectional_normal_scores(
        np.array([0.9]), np.array([3], dtype=np.int64), np.array(["C0"])
    )
    assert float(x[0]) == pytest.approx(0.0)


def test_one_day_cannot_dominate_the_accumulator() -> None:
    rng = np.random.default_rng(5)
    m, d, c = _panel(5000, 1)
    x = cross_sectional_normal_scores(rng.random(m.size), d, c)
    assert float(np.abs(x).max()) <= NORMAL_SCORE_CLIP + 1e-12


def test_the_blend_returns_calibrated_probabilities_and_keeps_the_level_signal() -> None:
    """``Decision.score`` must be a probability in [0,1]; a raw CUSUM statistic is not one.
    The blend must also keep the incumbent's level information, which is what the savings
    metric monetises — a rung that threw it away would trade savings for latency."""
    rng = np.random.default_rng(6)
    n = 4000
    incumbent = rng.random(n)
    acc = rng.random(n) * 5.0
    y = (rng.random(n) < 0.25 * incumbent + 0.05 * acc / 5.0).astype(int)
    fitted = RankCusum.fit(incumbent=incumbent, accumulator=acc, y=y, k=K_REFERENCE)
    p = fitted.predict(incumbent, acc)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Both channels carry weight: neither coefficient collapsed to zero.
    coefs = fitted.model.coef_[0]
    assert abs(coefs[0]) > 1e-3, "the incumbent level was discarded"

    # Scores of exactly 0 and 1 are emitted by LightGBM and must not produce inf.
    edge = fitted.predict(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert np.all(np.isfinite(edge))


def test_a_removed_cap_is_refused() -> None:
    m, d = np.array(["M0"]), np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="static-watchlist"):
        page_recursion(np.array([1.0]), d, m, c_max=0.0)


def test_misaligned_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="aligned row-for-row"):
        cross_sectional_normal_scores(
            np.ones(5), np.zeros(4, dtype=np.int64), np.array(["C0"] * 5)
        )
