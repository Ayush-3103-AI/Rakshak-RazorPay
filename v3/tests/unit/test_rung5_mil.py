"""T-0120 - Rung 5, multiple-instance learning by pooling over payer capsules.

Nothing here is scored on the real panel. ``data/v2/features.parquet`` is mid-regeneration
at a new geometry and the file on disk is a stale cycle-1 panel, so any number measured
against it would be a number about the wrong dataset. Every fixture below is constructed
in-process and is tens of merchants wide. The split-level result is the lead's, later.

What is asserted is everything that must hold regardless of how the rung scores, and the
weight of the file sits on two things:

**The pooling is provably the family it claims to be.** ``tau -> 0`` is mean-pooling and
``tau -> inf`` is max-pooling are asserted twice each - once as an exact identity at the
endpoint the grid actually contains, and once as convergence from the interior, because a
special-cased endpoint that agrees with nothing near it is a special case and not a limit.

**Stability is asserted against the naive form as a negative control.** "It is written in
log space" is a claim about arithmetic, and arithmetic can be run. Each stability test
computes the printed form beside the implemented one and asserts the printed form
*actually fails* - overflows to ``nan``, or underflows to ``0.0`` - so the test cannot
pass vacuously on a machine where the naive form happened to be fine.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rakshak.eval.metrics import pr_auc
from rakshak.explain.registry import Scorer
from rakshak.models.rung2_lgbm import DEFAULT_PARAMS
from rakshak.models.rung5_mil import (
    DEFAULT_TAU_GRID,
    EMPTY_BAG_SCORE,
    TrainedMIL,
    bag_offsets,
    feature_columns,
    fit_tau,
    pool,
    train,
)

#: Small on purpose: a feature-materialisation job owns this box's memory.
SMALL_PARAMS = replace(
    DEFAULT_PARAMS, n_estimators=60, num_leaves=7, min_child_samples=20, num_threads=1
)


def _ragged(sizes: list[int]) -> np.ndarray:
    """``bag_index`` for bags of the given sizes, in order."""
    return np.repeat(np.arange(len(sizes)), sizes).astype(np.intp)


# ─────────────────────────────────────────────────────────────────────────────
# The family: tau -> 0 is mean, tau -> inf is max
# ─────────────────────────────────────────────────────────────────────────────


def test_tau_zero_is_exactly_mean_pooling() -> None:
    p = np.array([0.1, 0.7, 0.3, 0.9, 0.2, 0.4, 0.6])
    idx = _ragged([3, 4])
    expected = np.array([p[:3].mean(), p[3:].mean()])
    # allclose at 1e-15, not array_equal: it *is* the arithmetic mean, but ``reduceat``
    # sums left to right and ``np.mean`` sums pairwise, so the last bit can differ.
    np.testing.assert_allclose(pool(p, idx, 2, tau=0.0), expected, rtol=1e-15)


def test_tau_inf_is_exactly_max_pooling() -> None:
    p = np.array([0.1, 0.7, 0.3, 0.9, 0.2, 0.4, 0.6])
    idx = _ragged([3, 4])
    expected = np.array([p[:3].max(), p[3:].max()])
    np.testing.assert_array_equal(pool(p, idx, 2, tau=np.inf), expected)


@pytest.mark.parametrize("tau", [1e-3, 1e-5, 1e-7])
def test_small_tau_approaches_the_mean_at_the_rate_the_expansion_predicts(tau: float) -> None:
    """``tau == 0`` is a special-cased branch, so this is what makes it a *limit*.

    Asserted against the second-order expansion ``s = mean(p) + (tau/2) * var(p) + O(tau^2)``
    rather than against the mean itself, and on the **difference**, not on the score. That
    is deliberate and it is the sharp version: a formulation that has lost its low-order
    digits to cancellation still lands within 1e-8 of the mean and would pass a loose
    comparison, while getting the ``(tau/2) * var`` term wrong. The first draft of
    ``_segment_lse_mean`` did exactly that.
    """
    p = np.array([0.05, 0.55, 0.95, 0.4, 0.6])
    idx = _ragged([3, 2])
    mean = pool(p, idx, 2, tau=0.0)
    curvature = np.array([p[:3].var(), p[3:].var()]) * tau / 2.0
    np.testing.assert_allclose(pool(p, idx, 2, tau=tau) - mean, curvature, rtol=1e-3)


@pytest.mark.parametrize("tau", [1e3, 1e5, 1e7])
def test_large_tau_converges_to_the_max_from_the_interior(tau: float) -> None:
    """The gap is ``log(n)/tau`` exactly, so the tolerance is derived, not guessed."""
    p = np.array([0.05, 0.55, 0.95, 0.4, 0.6])
    idx = _ragged([3, 2])
    got = pool(p, idx, 2, tau=tau)
    ceiling = pool(p, idx, 2, tau=np.inf)
    assert np.all(got <= ceiling + 1e-12)
    np.testing.assert_allclose(got, ceiling, atol=np.log(3.0) / tau + 1e-12)


def test_pooling_a_constant_bag_returns_the_constant_at_every_tau() -> None:
    """LSE of a constant vector is that constant for any tau, exactly. A pooling that
    drifts here has an ``n`` in the wrong place, which no ranking metric would reveal."""
    p = np.full(6, 0.37)
    idx = _ragged([6])
    for tau in (0.0, 1e-8, 0.5, 7.0, 1e6, np.inf):
        np.testing.assert_allclose(pool(p, idx, 1, tau=tau), [0.37], rtol=1e-12, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Numerical stability, each against the naive form as a negative control
# ─────────────────────────────────────────────────────────────────────────────


def test_lse_survives_a_tau_the_printed_form_overflows_on() -> None:
    """All-near-one bag at a large tau. ``exp(tau * p)`` is ``exp(1e6)``."""
    p = np.array([1.0 - 1e-16, 1.0, 0.999999, 1.0])
    idx = _ragged([4])
    tau = 1e6

    got = pool(p, idx, 1, tau=tau)
    assert np.isfinite(got).all()
    assert 0.0 <= got[0] <= 1.0
    np.testing.assert_allclose(got, [p.max()], atol=np.log(4.0) / tau + 1e-12)

    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.log(np.mean(np.exp(tau * p))) / tau
    assert not np.isfinite(naive), "negative control did not fail; the test proves nothing"


def test_lse_survives_an_all_near_zero_bag_at_a_vanishing_tau() -> None:
    """The other end: ``tau -> 0`` on probabilities that are themselves near zero, where
    the printed form is ``log(1 + eps)/tau`` and cancellation eats every digit."""
    p = np.array([1e-12, 2e-12, 3e-12, 4e-12])
    idx = _ragged([4])
    for tau in (1e-8, 1e-4, 1.0):
        got = pool(p, idx, 1, tau=tau)
        assert np.isfinite(got).all()
        assert float(p.mean()) <= got[0] + 1e-24 <= float(p.max()) + 1e-24


def test_noisy_or_survives_a_bag_of_vanishing_probabilities() -> None:
    """Five instances at 1e-300 pool to 5e-300, not to zero. The printed form
    ``1 - prod(1 - p)`` underflows to exactly 0.0 and silently ranks this bag level with a
    bag of certain innocents."""
    p = np.full(5, 1e-300)
    idx = _ragged([5])

    got = pool(p, idx, 1, kind="noisy_or")
    np.testing.assert_allclose(got, [5e-300], rtol=1e-12)
    assert got[0] > pool(np.zeros(5), idx, 1, kind="noisy_or")[0]

    naive = 1.0 - np.prod(1.0 - p)
    assert naive == 0.0, "negative control did not fail; the test proves nothing"


def test_noisy_or_is_exactly_one_when_an_instance_is_certain() -> None:
    """``log1p(-1) = -inf`` has to flow through the sum and come back as 1.0, not nan."""
    p = np.array([0.0, 1.0, 0.5])
    idx = _ragged([3])
    with np.errstate(all="raise"):  # nothing here may raise a floating-point error
        got = pool(p, idx, 1, kind="noisy_or")
    np.testing.assert_array_equal(got, [1.0])


def test_noisy_or_climbs_with_bag_size_at_fixed_evidence() -> None:
    """Documented in the module and asserted here, because it is the reason LSE is the
    family and noisy-OR is only a comparator: identical per-payer evidence, different bag
    size, very different score. On a panel whose typologies move transaction volume, that
    makes part of the noisy-OR score a transaction counter."""
    small = pool(np.full(3, 0.02), _ragged([3]), 1, kind="noisy_or")[0]
    large = pool(np.full(200, 0.02), _ragged([200]), 1, kind="noisy_or")[0]
    assert small < 0.07 < 0.9 < large


# ─────────────────────────────────────────────────────────────────────────────
# Bag semantics
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["lse", "noisy_or"])
def test_one_strongly_positive_instance_outscores_an_otherwise_identical_bag(
    kind: str,
) -> None:
    """Acceptance criterion 2, on bags that differ in exactly one instance."""
    quiet = np.array([0.01, 0.02, 0.03, 0.02])
    loud = quiet.copy()
    loud[2] = 0.95
    p = np.concatenate([quiet, loud])
    idx = _ragged([4, 4])
    scores = pool(p, idx, 2, tau=5.0, kind=kind)  # type: ignore[arg-type]
    assert scores[1] > scores[0]


@pytest.mark.parametrize("tau", [0.0, 1.0, 25.0, np.inf])
def test_pooling_is_monotone_in_every_instance(tau: float) -> None:
    rng = np.random.default_rng(7)
    p = rng.random(12)
    idx = _ragged([5, 3, 4])
    base = pool(p, idx, 3, tau=tau)
    for i in range(p.size):
        lifted = p.copy()
        lifted[i] = min(1.0, lifted[i] + 0.05)
        assert np.all(pool(lifted, idx, 3, tau=tau) >= base - 1e-12)


@pytest.mark.parametrize("kind", ["lse", "noisy_or"])
def test_a_merchant_day_with_no_payers_scores_the_documented_constant(kind: str) -> None:
    """A zero-payer day is a real occurrence, not an error. Bags 1 and 3 have no
    instances at all and must still appear in the score vector, in place."""
    p = np.array([0.4, 0.6, 0.8])
    idx = np.array([0, 0, 2], dtype=np.intp)
    scores = pool(p, idx, 4, tau=2.0, kind=kind)  # type: ignore[arg-type]
    assert scores.shape == (4,)
    assert scores[1] == EMPTY_BAG_SCORE
    assert scores[3] == EMPTY_BAG_SCORE
    assert scores[0] > 0.0 and scores[2] > 0.0


@pytest.mark.parametrize("kind", ["lse", "noisy_or"])
@pytest.mark.parametrize("tau", [0.0, 3.0, np.inf])
def test_a_singleton_bag_pools_to_itself_exactly(kind: str, tau: float) -> None:
    """The fallback ``TrainedMIL.predict`` takes with no ``bag_index``. It has to be the
    identity, not an approximation, or the Scorer-shaped call path would quietly report a
    different number from the bag-shaped one."""
    p = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    idx = _ragged([1] * 5)
    np.testing.assert_allclose(
        pool(p, idx, 5, tau=tau, kind=kind), p, rtol=1e-12, atol=1e-12  # type: ignore[arg-type]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input validation at the seam
# ─────────────────────────────────────────────────────────────────────────────


def test_bag_offsets_refuses_ungrouped_instances() -> None:
    """Pooling an unsorted index would merge two merchants into one bag and return a
    plausible number. capsules_as_of returns sorted rows; this catches a later reorder."""
    with pytest.raises(ValueError, match="non-decreasing"):
        bag_offsets(np.array([0, 1, 0, 1], dtype=np.intp), 2)


def test_bag_offsets_refuses_an_index_outside_the_bag_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        bag_offsets(np.array([0, 1, 5], dtype=np.intp), 2)


def test_bag_offsets_handles_no_instances_at_all() -> None:
    starts, labels, counts = bag_offsets(np.zeros(0, dtype=np.intp), 3)
    assert starts.size == labels.size == counts.size == 0
    np.testing.assert_array_equal(
        pool(np.zeros(0), np.zeros(0, dtype=np.intp), 3), np.full(3, EMPTY_BAG_SCORE)
    )


def test_pool_refuses_a_score_that_is_not_a_probability() -> None:
    with pytest.raises(ValueError, match="probabilities, not logits"):
        pool(np.array([-1.2, 3.4]), _ragged([2]), 1)


def test_pool_refuses_a_negative_tau() -> None:
    with pytest.raises(ValueError, match="min-pooling"):
        pool(np.array([0.2, 0.4]), _ragged([2]), 1, tau=-1.0)


def test_pool_refuses_an_unknown_pooling() -> None:
    with pytest.raises(ValueError, match="unknown pooling"):
        pool(np.array([0.2, 0.4]), _ragged([2]), 1, kind="attention")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# The axis the fitted tau reports on
# ─────────────────────────────────────────────────────────────────────────────


def _axis_fixture(witness: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal instance probabilities for two populations, built so the two ends of the
    family give opposite answers.

    ``witness=True``  - a positive bag holds one guilty payer among innocents, and its
                        *mean* is lower than a negative bag's. Mean-pooling must lose.
    ``witness=False`` - every payer of a positive bag is mildly off, and no single one
                        stands out. Max-pooling must lose.
    """
    per_bag = 5
    positive = [0.9, 0.05, 0.05, 0.05, 0.05] if witness else [0.3] * per_bag
    negative = [0.3] * per_bag if witness else [0.9, 0.05, 0.05, 0.05, 0.05]
    p = np.array([*(positive * 10), *(negative * 40)])
    y = np.array([1] * 10 + [0] * 40)
    return p, _ragged([per_bag] * 50), y


def test_a_witness_population_fits_the_max_end_of_the_family() -> None:
    p, idx, y = _axis_fixture(witness=True)
    assert pr_auc(y, pool(p, idx, y.size, tau=np.inf)) == 1.0
    assert pr_auc(y, pool(p, idx, y.size, tau=0.0)) < 0.3


def test_a_diffuse_population_fits_the_mean_end_of_the_family() -> None:
    """Ticket #54's declared-in-advance outcome: a tau near zero is the finding that the
    bag adds nothing the register's mean was not already taking. Asserted as a fixture the
    grid must be able to *reach*, so the finding is measurable rather than merely
    admissible."""
    p, idx, y = _axis_fixture(witness=False)
    assert pr_auc(y, pool(p, idx, y.size, tau=0.0)) == 1.0
    assert pr_auc(y, pool(p, idx, y.size, tau=np.inf)) < 0.3


def test_the_grid_spans_both_ends_and_contains_the_exact_endpoints() -> None:
    assert DEFAULT_TAU_GRID[0] == 0.0
    assert np.isinf(DEFAULT_TAU_GRID[-1])
    assert list(DEFAULT_TAU_GRID) == sorted(DEFAULT_TAU_GRID)


# ─────────────────────────────────────────────────────────────────────────────
# End to end, on a synthetic bag panel
# ─────────────────────────────────────────────────────────────────────────────


def _panel(
    seed: int, n_bags: int = 220
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """A capsule-shaped instance matrix: 13 columns, ragged bags, one guilty payer in
    each positive bag. Deliberately tiny - this is a plumbing test, not a benchmark."""
    rng = np.random.default_rng(seed)
    columns = feature_columns()
    sizes = rng.integers(2, 7, size=n_bags).tolist()
    bag_y = (rng.random(n_bags) < 0.25).astype(np.int8)
    idx = _ragged(sizes)
    x = rng.normal(size=(idx.size, len(columns)))
    for bag in np.flatnonzero(bag_y):
        rows = np.flatnonzero(idx == bag)
        x[rows[0], 0] += 4.0  # the witness
    return x, idx, bag_y, columns


def test_train_then_fit_tau_produces_a_reportable_rung() -> None:
    x, idx, bag_y, columns = _panel(seed=42)
    model = train(x, idx, bag_y, columns, params=SMALL_PARAMS)
    tuned, table = fit_tau(model, x, idx, bag_y)

    assert tuned.rung == 5
    assert len(table) == len(DEFAULT_TAU_GRID) + 1  # every grid point plus noisy-OR
    assert {row["pooling"] for row in table} == {"lse", "noisy_or"}

    summary = tuned.summary()
    assert summary["tau"] == tuned.tau  # AC 4: the fitted tau is in the results row
    assert summary["pooling"] == tuned.pooling
    assert summary["n_train_bags"] == bag_y.size
    assert summary["n_train_positive_bags"] == int(bag_y.sum())

    scores = tuned.predict(x, columns, bag_index=idx, n_bags=bag_y.size)
    assert scores.shape == (bag_y.size,)
    assert np.isfinite(scores).all() and scores.min() >= 0.0 and scores.max() <= 1.0
    assert pr_auc(bag_y, scores) > 0.5  # better than chance on a fixture with real signal


def test_fit_tau_selects_the_best_row_and_prefers_the_simpler_claim_on_a_tie() -> None:
    """Ties go to the earliest row. The grid is ascending with noisy-OR last, so a tie
    reports the smaller tau and reports LSE - both of which claim less."""
    x, idx, bag_y, columns = _panel(seed=1)
    model = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=1)
    tuned, table = fit_tau(model, x, idx, bag_y)
    scored = [row for row in table if not np.isnan(row["pr_auc"])]
    best = max(float(row["pr_auc"]) for row in scored)
    chosen = next(
        row
        for row in table
        if row["pooling"] == tuned.pooling
        and (row["tau"] is None if np.isnan(tuned.tau) else row["tau"] == tuned.tau)
    )
    assert float(chosen["pr_auc"]) == best


def test_fit_tau_refuses_a_validation_split_with_one_class() -> None:
    x, idx, bag_y, columns = _panel(seed=3)
    model = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=1)
    with pytest.raises(ValueError, match="one class"):
        fit_tau(model, x, idx, np.zeros_like(bag_y))


def test_pass_two_moves_the_instance_model_and_tau_zero_makes_it_a_no_op() -> None:
    """The pass-2 responsibility is the softmax of ``tau * p`` within the bag, so at
    ``tau = 0`` every instance is equally responsible and the refit is the pass-1 fit
    again. That identity is what makes the family claim hold through *training* and not
    only through scoring."""
    x, idx, bag_y, columns = _panel(seed=11)
    one_pass = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=1)
    flat = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=2, tau=0.0)
    sharp = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=2, tau=25.0)

    p1 = one_pass.instance_probabilities(x, columns)
    np.testing.assert_allclose(flat.instance_probabilities(x, columns), p1, rtol=1e-12)
    assert not np.allclose(sharp.instance_probabilities(x, columns), p1)


def test_train_is_deterministic_at_a_fixed_seed() -> None:
    x, idx, bag_y, columns = _panel(seed=5)
    a = train(x, idx, bag_y, columns, params=SMALL_PARAMS)
    b = train(x, idx, bag_y, columns, params=SMALL_PARAMS)
    np.testing.assert_array_equal(
        a.instance_probabilities(x, columns), b.instance_probabilities(x, columns)
    )


def test_train_refuses_ungrouped_rows() -> None:
    x, idx, bag_y, columns = _panel(seed=9)
    order = np.argsort(np.random.default_rng(0).random(idx.size))
    with pytest.raises(ValueError, match="non-decreasing"):
        train(x[order], idx[order], bag_y, columns, params=SMALL_PARAMS)


def test_train_refuses_labels_that_are_not_bag_labels() -> None:
    x, idx, bag_y, columns = _panel(seed=13)
    with pytest.raises(ValueError, match="0/1 bag labels"):
        train(x, idx, bag_y.astype(np.float64) + 0.5, columns, params=SMALL_PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# The contract it plugs into
# ─────────────────────────────────────────────────────────────────────────────


def test_rung5_satisfies_the_scorer_protocol() -> None:
    """AC 3: it registers as a ``Scorer`` and needs no new seam. The import lives in the
    test rather than in the rung, because ``test_explain_registry.py`` asserts that
    ``models/`` does not import the explain package."""
    x, idx, bag_y, columns = _panel(seed=17)
    model = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=1)
    assert isinstance(model, Scorer)
    assert isinstance(model, TrainedMIL)


def test_predict_without_a_bag_index_scores_one_row_at_a_time() -> None:
    """The Scorer-shaped call. Every row is its own singleton bag, and a singleton pools
    to itself exactly, so this is the instance probability and not a degraded bag score."""
    x, idx, bag_y, columns = _panel(seed=19)
    model = train(x, idx, bag_y, columns, params=SMALL_PARAMS, passes=1)
    np.testing.assert_array_equal(
        model.predict(x, columns), model.instance_probabilities(x, columns)
    )


def test_the_instance_columns_are_the_capsule_vector_in_schema_order() -> None:
    from rakshak.features.capsules import CAPSULE_COLUMNS

    assert feature_columns() == CAPSULE_COLUMNS[4:]
    assert "payer_id" not in feature_columns()
