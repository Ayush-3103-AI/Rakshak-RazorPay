"""T-0011 — the FR-018 ablation plumbing, and the guarantee that it is additive.

Three things are pinned here, in order of how much damage they prevent:

1. **The new keyword arguments are behaviour-preserving at their defaults.** Every
   existing caller of `build_emissions` / `build_window_matrix` passes neither
   `drop_features` nor `standardise`, so a variant-free build must be bit-identical
   to the pre-T-0011 one. If that ever stops holding, every number in
   `results/summary.md` silently moves.
2. **Each ablation removes what it claims to remove** — exactly the four FR-008
   graph scalars, and the within-merchant z-scoring itself rather than some
   downstream proxy for it.
3. **The rendered table carries a row for every FR-018 component**, including the
   two that were never measured. A struck row that quietly vanishes is the failure
   mode FR-018 exists to prevent.

The rendering test uses hand-built rows rather than a real fit: it is checking the
document's completeness, not the numbers, and a 6-fit run does not belong in the
test suite (NFR-004).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak.config import GENERATOR_START_DATE, SEED
from rakshak.eval import ablations
from rakshak.eval.harness import MODEL_REGISTRY
from rakshak.eval.splits import Split
from rakshak.features import BASE_FEATURES, build_emissions, build_window_features

_METHODS = ("UPI", "CARD", "NETBANKING")


def _merchant(merchant_id: str, aov: float, mcc: str, seed: int) -> pd.DataFrame:
    """One merchant's stream: 12 windows x 20 transactions, ticket scale ``aov`` INR."""
    rng = np.random.default_rng(seed)
    n_windows, per_window = 12, 20
    n = n_windows * per_window
    day = np.repeat(np.arange(n_windows) * 7, per_window) + rng.integers(0, 7, n)
    return pd.DataFrame(
        {
            "merchant_id": merchant_id,
            "timestamp": pd.Timestamp(GENERATOR_START_DATE)
            + pd.to_timedelta(day, unit="D")
            + pd.to_timedelta(rng.integers(8, 22, n), unit="h"),
            "amount": np.round(aov * np.exp(rng.normal(0.0, 0.3, n)), 2),
            "payer_id": [f"{merchant_id}-P{p}" for p in rng.integers(0, 30, n)],
            "method": rng.choice(_METHODS, n),
            "mcc": mcc,
            "is_refund": rng.random(n) < 0.05,
            "is_chargeback": rng.random(n) < 0.02,
        }
    ).sort_values("timestamp", kind="stable")


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    """A three-merchant panel wide enough to standardise and long enough to window."""
    return pd.concat(
        [
            _merchant("M1", aov=120.0, mcc="5411", seed=SEED),
            _merchant("M2", aov=9000.0, mcc="5944", seed=SEED + 1),
            _merchant("M3", aov=400.0, mcc="5411", seed=SEED + 2),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# 1. The defaults are behaviour-preserving
# ---------------------------------------------------------------------------


def test_default_build_is_identical_to_a_variant_free_build(
    transactions: pd.DataFrame,
) -> None:
    """Passing the new arguments at their defaults must change nothing, bitwise."""
    shipping = build_emissions(transactions)
    explicit = build_emissions(transactions, drop_features=(), standardise=True)

    np.testing.assert_array_equal(shipping.X, explicit.X)
    assert shipping.feature_names == explicit.feature_names
    np.testing.assert_array_equal(shipping.merchant_ids, explicit.merchant_ids)
    np.testing.assert_array_equal(shipping.shrinkage_weight, explicit.shrinkage_weight)


def test_ablations_module_leaves_the_model_registry_untouched() -> None:
    """A variant is registered only for the duration of its own scoring pass."""
    before = dict(MODEL_REGISTRY)

    def _boom(split: Split, rng: np.random.Generator) -> pd.Series:  # pragma: no cover
        raise AssertionError("never called")

    with ablations._registered("variant-under-test", _boom):
        assert "variant-under-test" in MODEL_REGISTRY
    assert MODEL_REGISTRY == before

    # ... and also when the scoring pass raises.
    with pytest.raises(RuntimeError), ablations._registered("variant-under-test", _boom):
        raise RuntimeError("scoring blew up")
    assert MODEL_REGISTRY == before


# ---------------------------------------------------------------------------
# 2. Each ablation removes what it claims to remove
# ---------------------------------------------------------------------------


def test_dropping_graph_features_removes_exactly_those_four_columns(
    transactions: pd.DataFrame,
) -> None:
    """FR-008 / ADR-0002: the four graph scalars go, and nothing else moves."""
    shipping = build_emissions(transactions)
    reduced = build_emissions(transactions, drop_features=ablations.GRAPH_FEATURES)

    assert set(ablations.GRAPH_FEATURES) <= set(BASE_FEATURES)
    assert set(shipping.feature_names) - set(reduced.feature_names) == set(
        ablations.GRAPH_FEATURES
    )
    assert reduced.X.shape[2] == shipping.X.shape[2] - len(ablations.GRAPH_FEATURES)

    # The surviving columns are the same numbers in the same order: dropping a
    # feature must not perturb the standardisation of the ones that remain.
    kept = [shipping.feature_names.index(name) for name in reduced.feature_names]
    np.testing.assert_array_equal(reduced.X, shipping.X[:, :, kept])


def test_unknown_drop_feature_is_rejected(transactions: pd.DataFrame) -> None:
    """A typo in an ablation must fail loudly, not silently drop nothing."""
    with pytest.raises(ValueError, match="do not exist"):
        build_emissions(transactions, drop_features=("no_such_feature",))


def test_standardise_off_bypasses_the_z_scoring(transactions: pd.DataFrame) -> None:
    """FR-007 off means the raw per-window aggregates, not a rescaled version of them."""
    panel, feature_names = build_window_features(transactions)
    raw = build_emissions(transactions, standardise=False)

    expected = panel[list(feature_names)].to_numpy(dtype=float).reshape(raw.X.shape)
    np.testing.assert_allclose(raw.X, expected, rtol=0.0, atol=0.0)

    # And the shipping path is genuinely different: the two merchants that differ only
    # in ticket scale are near-identical after standardisation (FR-007's own acceptance
    # test) and orders of magnitude apart before it.
    amount_column = feature_names.index("log_amount_mean")
    standardised = build_emissions(transactions)
    assert np.ptp(raw.X[:, :, amount_column]) > np.ptp(
        standardised.X[:, :, amount_column]
    )


# ---------------------------------------------------------------------------
# 3. The rendered table is complete, struck rows included
# ---------------------------------------------------------------------------


def _row(savings: float, pr_auc: float) -> dict[str, object]:
    """A minimal `harness.evaluate_model` row, enough to render the table."""
    return {
        "savings": savings,
        "pr_auc": pr_auc,
        "precision_at_k": 0.5,
        "brier": 0.2,
        "lag_days": 1.0,
        "flagged_fraction": 0.5,
        "n_reviewed": 5,
        "n_held": 10,
        "hours_used": 0.33,
        "binding_constraint": "capacity",
        "unconstrained_n_reviewed": 9,
    }


@pytest.fixture()
def rendered() -> str:
    """Render the document from hand-built rows — completeness, not numbers."""
    merchant_ids = tuple(f"M{i}" for i in range(10))
    index = pd.Index(merchant_ids, name="merchant_id")
    labels = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=index)
    split = Split(
        name="test",
        start_day=210,
        end_day=270,
        merchant_ids=merchant_ids,
        transactions=pd.DataFrame(),
        labels=labels,
        transition_day=pd.Series(np.nan, index=index),
        transition_timestamp=pd.Series(pd.NaT, index=index),
        loss_inr=pd.Series([5000.0, 4000.0] + [0.0] * 8, index=index),
        value_inr=pd.Series(1000.0, index=index),
    )
    keys = [config.key for config in ablations.CONFIGS] + list(ablations.CONTEXT_MODELS)
    rows = {key: _row(0.5 + 0.01 * i, 0.3 + 0.01 * i) for i, key in enumerate(keys)}
    return ablations.render_ablations(split, rows, seed=SEED, k=5, capacity_hours=0.4)


@pytest.mark.parametrize(
    "component",
    [
        "HMM (the proposal)",
        "graph features (FR-008)",
        "within-merchant standardisation (FR-007)",
        "empirical-Bayes shrinkage (ADR-0006)",
        "NSGA-II vs. grid search (ADR-0004)",
    ],
)
def test_every_fr018_component_has_a_row(rendered: str, component: str) -> None:
    """FR-018 names five components; all five appear, measured or not."""
    assert f"| {component} |" in rendered


def test_struck_rows_render_as_not_measured_never_as_zero(rendered: str) -> None:
    """A cut ticket's row must say so. Zero and absent both make a false claim."""
    struck = [
        line
        for line in rendered.splitlines()
        if line.startswith(("| empirical-Bayes shrinkage", "| NSGA-II vs. grid search"))
        and ablations._NOT_MEASURED in line
    ]
    assert len(struck) == 2
    for line in struck:
        cells = [cell.strip() for cell in line.strip("|").split("|")][2:]
        assert cells and all(cell == ablations._NOT_MEASURED for cell in cells)
    assert "T-0008 was cut" in rendered
    assert "T-0009 was cut" in rendered


def test_the_open_question_about_bocpd_is_stated_as_open(rendered: str) -> None:
    """T-0010 was cut, so no other sequence-aware baseline exists. Say so."""
    assert "T-0010 (BOCPD" in rendered
    assert "left open, and it is stated as open" in rendered


def test_no_configuration_was_selected_on_test(rendered: str) -> None:
    """The unlock is only honest if the document says what it did not do."""
    assert "No configuration was selected on `test`" in rendered
    assert "T-0004b on `validate`" in rendered


def test_every_savings_number_is_printed_beside_a_pr_auc(rendered: str) -> None:
    """AP-06: a savings column without PR-AUC beside it is not interpretable."""
    for line in rendered.splitlines():
        if line.startswith("| component |") or line.startswith("| model |"):
            header = [cell.strip() for cell in line.strip("|").split("|")]
            assert "savings" in header and "PR-AUC" in header
