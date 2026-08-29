"""T-0004 — the Saturday EOD gate: HMM state recovery on REAL generator output.

T-0002 answered kill criterion K1 on a throwaway toy fixture with three balanced states
separated by 3 sigma. This module asks the same question of T-0003's full 747k-row
merchant population passed through T-0004's feature layer. That is the number that
decides whether the sequence layer earns its place.

WHAT THIS MODULE RECORDS
------------------------
It records a FAILURE, honestly, per CLAUDE.md non-negotiable #1. FR-013's ARI > 0.5 gate
does NOT hold at full scale. The gate test is therefore `xfail(strict=True)`: it runs, it
fails, the suite stays green, and the moment anyone makes it pass the strict flag turns
that into an error so the finding cannot rot silently.

Crucially, `test_oracle_ceiling_is_below_the_gate` proves this is not an optimisation
failure that a better initialiser would fix. Setting the HMM's parameters directly from
the ground-truth labels — the best a 4-state diagonal-Gaussian HMM could possibly do on
these emissions — still lands below 0.5. The emission geometry is the binding constraint,
not Baum-Welch.

The leading indicator is `test_report_per_typology_separation`, carried forward from
T-0002's sweep as an acceptance requirement: separation in sigma predicts where the model
is blind, and it does so before ARI confirms it.

Numbers here are measured on SYNTHETIC merchant streams with injected typologies; the
generator is in this repo (CLAUDE.md non-negotiable #3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from rakshak.cli import seed_everything
from rakshak.config import (
    ARI_RECOVERY_THRESHOLD,
    N_HIDDEN_STATES,
    SEED,
    SPLIT_DAY_BOUNDS,
    STATE_PATHS_PARQUET,
    TRANSACTIONS_PARQUET,
    WINDOW_DAYS,
)
from rakshak.eval.metrics import (
    align_states,
    detection_lag_windows,
    state_recovery_report,
)
from rakshak.eval.splits import assign_merchant_groups
from rakshak.features import (
    MIN_SEGMENT_MERCHANTS,
    EmissionSet,
    build_emissions,
    window_state_labels,
)
from rakshak.models.hmm import HMM, LABEL_CLAMP_LOG, UNLABELLED

STATE_ORDER: tuple[str, ...] = ("HEALTHY", "RAMP", "FRAUD", "DORMANT")


@dataclass(frozen=True)
class FullScale:
    """Everything the gate needs, computed once.

    Attributes:
        emissions: Standardised emissions, ``X`` of shape (M, W, D).
        labels: Ground-truth state names, shape (M, W).
        codes: `labels` as indices into `STATE_ORDER`, shape (M, W).
        typology: Injected typology per merchant, shape (M,).
        decoded: Viterbi state indices from the fitted HMM, shape (M, W).
        history: Baum-Welch log-likelihood per EM iteration, nats.
        model: The fitted HMM itself, so a caller can ask it for the posterior rather
            than scoring a threshold-free metric on an already-thresholded decode.
    """

    emissions: EmissionSet
    labels: np.ndarray
    codes: np.ndarray
    typology: np.ndarray
    decoded: np.ndarray
    history: list[float]
    model: HMM


@pytest.fixture(scope="module")
def full_scale() -> FullScale:
    """Build emissions from the real generator output and fit one pooled HMM.

    Pooled across the whole population rather than per segment: a per-segment fit was
    measured during T-0004 and scored materially worse (ARI 0.02 vs 0.09), because each
    segment holds only ~24 merchants and roughly two of them are ever non-healthy.
    """
    if not TRANSACTIONS_PARQUET.exists() or not STATE_PATHS_PARQUET.exists():
        pytest.skip("run `python -m rakshak.generator.generate --seed 42` first")

    transactions = pd.read_parquet(TRANSACTIONS_PARQUET)
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    emissions = build_emissions(transactions)
    labels = window_state_labels(state_paths, emissions.merchant_ids, emissions.X.shape[1])
    codes = np.vectorize(STATE_ORDER.index)(labels)
    typology = (
        state_paths.groupby("merchant_id")["typology"]
        .first()
        .reindex(emissions.merchant_ids)
        .to_numpy()
    )

    rng = seed_everything(SEED)
    model = HMM(n_states=N_HIDDEN_STATES, n_features=emissions.X.shape[2])
    history = model.fit(emissions.sequences(), rng=rng)
    decoded = np.stack([model.viterbi(x) for x in emissions.sequences()])
    return FullScale(emissions, labels, codes, typology, decoded, history, model)


def separation_sigma(emissions: EmissionSet, labels: np.ndarray, row: int) -> float:
    """Distance from one merchant's healthy baseline to its non-healthy windows, in sigma.

    The diagonal Mahalanobis distance between the two window groups, using the merchant's
    OWN healthy windows as the reference scale. Directly comparable to T-0002's state
    separation sweep, which found ARI 0.41 at 0.75 sigma and 0.79 at 1.5 sigma.

    Args:
        emissions: The emission set.
        labels: Ground-truth state names, shape (M, W).
        row: Merchant row index.

    Returns:
        Separation in standard deviations, or NaN when the merchant has too few windows
        of either kind to estimate one.
    """
    healthy = labels[row] == "HEALTHY"
    if healthy.sum() < 3 or (~healthy).sum() < 2:
        return float("nan")
    baseline = emissions.X[row, healthy]
    scale = baseline.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    delta = (emissions.X[row, ~healthy].mean(axis=0) - baseline.mean(axis=0)) / scale
    return float(np.linalg.norm(delta))


def oracle_hmm(emissions: EmissionSet, codes: np.ndarray) -> HMM:
    """An HMM whose parameters are read straight off the ground truth.

    Not a model — a measuring instrument. It answers "what is the best a 4-state
    diagonal-Gaussian HMM could do on these emissions", which is the only way to tell a
    bad optimum apart from an unrecoverable one.

    Args:
        emissions: The emission set.
        codes: Ground-truth state indices, shape (M, W).

    Returns:
        An `HMM` with maximum-likelihood parameters under the true labels.
    """
    flat = emissions.X.reshape(-1, emissions.X.shape[2])
    flat_codes = codes.ravel()
    model = HMM(n_states=len(STATE_ORDER), n_features=emissions.X.shape[2])
    for k in range(len(STATE_ORDER)):
        members = flat[flat_codes == k]
        model.mu[k] = members.mean(axis=0)
        model.var[k] = members.var(axis=0) + 1e-6

    initial = np.bincount(codes[:, 0], minlength=len(STATE_ORDER)) + 1e-8
    model.log_pi = np.log(initial / initial.sum())
    counts = np.zeros((len(STATE_ORDER), len(STATE_ORDER))) + 1e-8
    np.add.at(counts, (codes[:, :-1].ravel(), codes[:, 1:].ravel()), 1.0)
    model.log_A = np.log(counts / counts.sum(axis=1, keepdims=True))
    return model


# ------------------------------------------------------------------------------------
# Sanity: the pipeline is wired up correctly before any of its numbers mean anything
# ------------------------------------------------------------------------------------


def test_emission_panel_shape_and_alignment(full_scale: FullScale) -> None:
    """Ground truth and emissions must share one grid, or ARI measures an offset."""
    emissions = full_scale.emissions
    assert emissions.X.ndim == 3
    assert full_scale.labels.shape == emissions.X.shape[:2]
    assert full_scale.decoded.shape == emissions.X.shape[:2]
    assert emissions.n_txn.shape == emissions.X.shape[:2]
    assert np.isfinite(emissions.X).all()
    # Nothing has gone wrong yet during any merchant's burn-in, so the standardisation
    # baseline is uncontaminated — the other half of FR-007's leakage guard.
    assert (full_scale.labels[:, : emissions.burn_in_windows] == "HEALTHY").all()


def test_every_segment_meets_the_floor_on_the_real_population(full_scale: FullScale) -> None:
    """FR-011 on the population we actually train on, not on a constructed example."""
    counts = pd.Series(full_scale.emissions.segments).value_counts()
    assert counts.min() >= MIN_SEGMENT_MERCHANTS, counts.to_dict()


def test_baum_welch_stays_monotone_on_real_features(full_scale: FullScale) -> None:
    """A decreasing log-likelihood is always an M-step bug (07-math.md §1).

    This passing while ARI fails is precisely the quiet failure T-0002 warned about:
    clean monotone convergence, plausible Viterbi path, pointing at nothing.
    """
    history = full_scale.history
    assert len(history) >= 3
    assert (np.diff(history) >= -1e-6).all(), f"log-likelihood decreased: {history}"


# ------------------------------------------------------------------------------------
# The leading indicator — measured and reported BEFORE the ARI verdict
# ------------------------------------------------------------------------------------


def test_report_per_typology_separation(full_scale: FullScale) -> None:
    """Report per-typology feature separation in sigma. The leading indicator.

    Asserts only the floor that makes the diagnostic meaningful — that the four
    non-adversarial typologies are separated at all. SLOW_RAMP is deliberately excluded
    from any assertion: it exists to be reported as a failure mode (FR-005), and tuning
    the feature layer until it clears a threshold would be rigging the exam.
    """
    lines = ["", "per-typology separation from the merchant's own healthy baseline (sigma)"]
    lines.append(f"{'typology':22s} {'n':>3s} {'median':>7s} {'p25':>7s} {'p75':>7s}")
    measured: dict[str, float] = {}
    for typ in ("BUST_OUT", "LAUNDERING_ENDPOINT", "CATEGORY_DRIFT", "REFUND_COLLUSION",
                "SLOW_RAMP"):
        rows = np.flatnonzero(full_scale.typology == typ)
        values = np.array(
            [separation_sigma(full_scale.emissions, full_scale.labels, r) for r in rows]
        )
        values = values[np.isfinite(values)]
        measured[typ] = float(np.median(values))
        lines.append(
            f"{typ:22s} {len(values):3d} {np.median(values):7.2f} "
            f"{np.percentile(values, 25):7.2f} {np.percentile(values, 75):7.2f}"
        )
    print("\n".join(lines))

    for typ in ("BUST_OUT", "LAUNDERING_ENDPOINT", "CATEGORY_DRIFT", "REFUND_COLLUSION"):
        assert measured[typ] > 1.0, (
            f"{typ} separates at only {measured[typ]:.2f} sigma; T-0002's sweep puts the "
            "ARI > 0.5 gate at roughly 1.0 sigma, so the feature layer is the problem"
        )


def test_report_per_state_separation(full_scale: FullScale) -> None:
    """The separation ARI actually depends on: each latent state against HEALTHY.

    Per-typology separation is strong and per-state separation is not, and the gap
    between those two facts is the whole finding of T-0004. A typology is detectable;
    the four-way state partition that FR-013 scores is a different question.
    """
    flat = full_scale.emissions.X.reshape(-1, full_scale.emissions.X.shape[2])
    codes = full_scale.codes.ravel()
    healthy = flat[codes == 0]
    scale = np.where(healthy.std(axis=0) > 1e-6, healthy.std(axis=0), 1.0)

    separations: dict[str, float] = {}
    for k, name in enumerate(STATE_ORDER):
        if k == 0:
            continue
        delta = (flat[codes == k].mean(axis=0) - healthy.mean(axis=0)) / scale
        separations[name] = float(np.linalg.norm(delta))
        print(f"{name:9s} vs HEALTHY: {separations[name]:6.2f} sigma  n={int((codes == k).sum())}")

    assert separations["DORMANT"] > 5.0
    assert separations["FRAUD"] > 2.0
    # RAMP is the one that is not separable, and that is the reported finding.
    assert separations["RAMP"] < 2.0, (
        "RAMP now separates better than when T-0004 measured it; re-check whether the "
        "gate below has become reachable"
    )


# ------------------------------------------------------------------------------------
# The gate — and the proof that failing it is not an optimisation problem
# ------------------------------------------------------------------------------------


def test_oracle_ceiling_is_below_the_gate(full_scale: FullScale) -> None:
    """The best any 4-state diagonal-Gaussian HMM could do here is still under 0.5.

    Parameters are set from the ground-truth labels, so Baum-Welch is removed from the
    picture entirely. This is what makes the gate failure a finding about the emission
    geometry rather than a bug report about EM, and it is why the honest response is a
    DESCEND rather than a round of hyperparameter tuning.
    """
    model = oracle_hmm(full_scale.emissions, full_scale.codes)
    decoded = np.stack([model.viterbi(x) for x in full_scale.emissions.sequences()])
    ceiling = adjusted_rand_score(full_scale.codes.ravel(), decoded.ravel())
    print(f"\noracle-parameterised ARI ceiling = {ceiling:.3f}")
    assert ceiling < ARI_RECOVERY_THRESHOLD, (
        f"the ceiling has risen to {ceiling:.3f}; the gate may now be reachable by "
        "fitting, and T-0004's DESCEND recommendation should be revisited"
    )


def test_report_ari_with_and_without_slow_ramp(full_scale: FullScale) -> None:
    """Both numbers, always. Excluding SLOW_RAMP must never be how the gate is cleared.

    It does not clear it here either — which is itself the important part. The failure is
    broader than the adversarial typology.
    """
    codes, decoded, typology = full_scale.codes, full_scale.decoded, full_scale.typology
    ari_all = adjusted_rand_score(codes.ravel(), decoded.ravel())
    keep = typology != "SLOW_RAMP"
    ari_without = adjusted_rand_score(codes[keep].ravel(), decoded[keep].ravel())
    print(f"\nARI all merchants          = {ari_all:.3f}")
    print(f"ARI excluding SLOW_RAMP    = {ari_without:.3f}")
    for typ in ("NONE", "BUST_OUT", "LAUNDERING_ENDPOINT", "CATEGORY_DRIFT",
                "REFUND_COLLUSION", "SLOW_RAMP"):
        rows = typology == typ
        if rows.any():
            score = adjusted_rand_score(codes[rows].ravel(), decoded[rows].ravel())
            print(f"   ARI[{typ:20s}] n={int(rows.sum()):3d}  {score:.3f}")

    assert ari_without < ARI_RECOVERY_THRESHOLD or ari_all > ARI_RECOVERY_THRESHOLD, (
        "dropping SLOW_RAMP is what carried the gate; report both numbers and do not "
        "let the adversarial typology be quietly excluded (CLAUDE.md non-negotiable #1)"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "K1 FIRES AND STILL FIRES AFTER THE T-0004b REMEDIATION. FR-013's original "
        "ARI > 0.5 does not hold on full-scale realistic features and is retired as "
        "unreachable by the dated amendment at FR-013 in 06-requirements.md. Measured at "
        "seed 42: unsupervised pooled ARI 0.147 (0.157 excluding SLOW_RAMP) against an "
        "oracle-parameterised ceiling of 0.404 — the ceiling is BELOW the gate, so this is "
        "the emission geometry, not Baum-Welch. Label-informed partially-supervised fitting "
        "(T-0004b item 2) raises four-way ARI to 0.319 and AMI to 0.218 on the never-labelled "
        "validate group, i.e. ~85% of the way to the ceiling, and still nowhere near 0.5. "
        "RAMP sits 1.19 sigma from HEALTHY and HEALTHY is 91% of all windows. See "
        "logbook-entries/T-0004.md and logbook-entries/T-0004b.md; this xfail is strict so "
        "the finding cannot rot unnoticed."
    ),
)
def test_state_recovery_ari_full_scale(full_scale: FullScale) -> None:
    """THE GATE (FR-013). Recorded as a known failure, not weakened and not hidden."""
    ari = adjusted_rand_score(full_scale.codes.ravel(), full_scale.decoded.ravel())
    assert ari > ARI_RECOVERY_THRESHOLD, (
        f"full-scale ARI {ari:.3f} <= {ARI_RECOVERY_THRESHOLD} (FR-013) — kill criterion K1"
    )


# ------------------------------------------------------------------------------------
# T-0004b — label-weighted partially-supervised fitting, and the amended FR-013 suite
# ------------------------------------------------------------------------------------
#
# ADR-0005 / the FR-013 amendment dated 2026-08-28 in 06-requirements.md. The estimation
# gap (unsupervised 0.147 -> supervised-MLE ceiling 0.404) is closable; the representation
# gap (0.404 -> 1.0) is not. What follows measures the first and reports the second.
#
# EVERY number below is scored on the VALIDATE merchant group, whose labels are never
# handed to the fit and whose windows are never clamped. The "all windows" number is
# printed too, flagged as contaminated, because 7500 of its 19500 windows had their
# posterior clamped to the truth and it would be dishonest to quote it as recovery.


def train_label_grid(emissions: EmissionSet, codes: np.ndarray, state_paths: pd.DataFrame):
    """Build the (M, W) label grid handed to `HMM.fit_partial`, and the group masks.

    A window may carry its ground-truth label only if BOTH split dimensions allow it
    (06-requirements.md §3): the merchant is in the training group, AND the window ends
    before the training temporal window does. Everything else is `UNLABELLED`.

    Args:
        emissions: The emission set.
        codes: Ground-truth state indices, shape (M, W).
        state_paths: The generator's state-path frame, for group assignment.

    Returns:
        `(labels, is_train, is_validate, window_in_train)`. `labels` is int64 (M, W).
    """
    groups = assign_merchant_groups(state_paths).reindex(emissions.merchant_ids)
    is_train = groups.to_numpy() == "train"
    is_validate = groups.to_numpy() == "validate"
    n_windows = codes.shape[1]
    train_end_day = SPLIT_DAY_BOUNDS["train"][1]
    window_in_train = (np.arange(n_windows) * WINDOW_DAYS + WINDOW_DAYS) <= train_end_day

    labels = np.full(codes.shape, UNLABELLED, dtype=np.int64)
    allowed = is_train[:, None] & window_in_train[None, :]
    labels[allowed] = codes[allowed]
    return labels, is_train, is_validate, window_in_train


@pytest.fixture(scope="module")
def partial_fit(full_scale: FullScale) -> tuple[HMM, np.ndarray, np.ndarray]:
    """Fit the label-weighted partially-supervised HMM on TRAINING labels only.

    Returns:
        `(model, labels, decoded)` — the fitted model, the label grid it was given, and
        its Viterbi decode over every merchant.
    """
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    labels, _, _, _ = train_label_grid(full_scale.emissions, full_scale.codes, state_paths)
    rng = seed_everything(SEED)
    model = HMM(n_states=N_HIDDEN_STATES, n_features=full_scale.emissions.X.shape[2])
    model.fit_partial(
        full_scale.emissions.sequences(), [labels[i] for i in range(labels.shape[0])], rng=rng
    )
    decoded = np.stack([model.viterbi(x) for x in full_scale.emissions.sequences()])
    return model, labels, decoded


# --- The leakage guard. A leaked number is worse than a failed gate. -------------------


def test_labels_reach_only_the_training_split(full_scale: FullScale) -> None:
    """No validate or test label may enter the fitting path. NFR-002, zero tolerance."""
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    labels, is_train, is_validate, window_in_train = train_label_grid(
        full_scale.emissions, full_scale.codes, state_paths
    )
    assert (labels[~is_train] == UNLABELLED).all(), "a non-training merchant carries a label"
    assert (labels[:, ~window_in_train] == UNLABELLED).all(), "a post-train window is labelled"
    assert (labels[is_validate] == UNLABELLED).all(), "a validate merchant carries a label"
    labelled = labels != UNLABELLED
    assert labelled.any(), "nothing is labelled; the test would pass vacuously"
    assert (labelled == (is_train[:, None] & window_in_train[None, :])).all()
    print(
        f"\nlabelled windows {int(labelled.sum())} of {labels.size} "
        f"({int(is_train.sum())} training merchants x {int(window_in_train.sum())} windows)"
    )


def test_corrupting_heldout_labels_does_not_move_the_fit(
    full_scale: FullScale, partial_fit: tuple[HMM, np.ndarray, np.ndarray]
) -> None:
    """The end-to-end leakage proof: scramble every held-out label, refit, get the same model.

    A structural assertion that the grid is -1 outside training can be satisfied by a grid
    builder that is right today and wrong after an edit. This one cannot: if any validate
    or test information reached the estimator by any route, these parameters would differ.
    """
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    model, _, _ = partial_fit

    rng_corrupt = np.random.default_rng(999)
    corrupted = full_scale.codes.copy()
    _, is_train, _, _ = train_label_grid(full_scale.emissions, full_scale.codes, state_paths)
    held_out = ~is_train
    corrupted[held_out] = rng_corrupt.integers(
        0, len(STATE_ORDER), size=corrupted[held_out].shape
    )
    labels, _, _, _ = train_label_grid(full_scale.emissions, corrupted, state_paths)

    rng = seed_everything(SEED)
    refit = HMM(n_states=N_HIDDEN_STATES, n_features=full_scale.emissions.X.shape[2])
    refit.fit_partial(
        full_scale.emissions.sequences(), [labels[i] for i in range(labels.shape[0])], rng=rng
    )
    for name in ("mu", "var", "log_A", "log_pi"):
        np.testing.assert_array_equal(
            getattr(refit, name), getattr(model, name), err_msg=f"{name} moved — LABEL LEAK"
        )


def test_unlabelled_partial_fit_is_exactly_unsupervised() -> None:
    """`fit_partial` with nothing labelled must reproduce `fit` bit-for-bit.

    This is what makes the leakage argument airtight from the model's side: an unlabelled
    window contributes to the M-step exactly as it does under plain Baum-Welch, so held-out
    windows cannot influence the fit through the label pathway at all.
    """
    rng = np.random.default_rng(7)
    sequences = [rng.normal(size=(24, 3)) for _ in range(6)]
    blank = [np.full(24, UNLABELLED, dtype=np.int64) for _ in sequences]

    a = HMM(n_states=3, n_features=3)
    a.fit(sequences, rng=np.random.default_rng(1))
    b = HMM(n_states=3, n_features=3)
    b.fit_partial(sequences, blank, rng=np.random.default_rng(1))
    for name in ("mu", "var", "log_A", "log_pi"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_clamped_posterior_matches_the_label() -> None:
    """A labelled timestep's smoothed posterior must sit on the state it was given."""
    rng = np.random.default_rng(3)
    sequences = [rng.normal(size=(30, 2)) for _ in range(5)]
    labels = [np.full(30, UNLABELLED, dtype=np.int64) for _ in sequences]
    labels[0][:10] = 1
    model = HMM(n_states=2, n_features=2)
    model.fit_partial(sequences, labels, rng=np.random.default_rng(2))

    log_b = model.log_emission(sequences[0])
    clamp = np.zeros_like(log_b)
    clamp[:10, 0] = LABEL_CLAMP_LOG
    log_alpha, ll = model._forward_logb(log_b + clamp)
    log_beta = model._backward_logb(log_b + clamp)
    gamma = np.exp(log_alpha + log_beta - ll)
    assert gamma[:10, 1].min() > 1.0 - 1e-9, gamma[:10]


# --- The amended FR-013 suite, reported in full ---------------------------------------


def test_report_amended_fr013_suite(
    full_scale: FullScale, partial_fit: tuple[HMM, np.ndarray, np.ndarray]
) -> None:
    """Print the amended FR-013 suite for baseline, label-informed model and oracle ceiling.

    ARI and the oracle ceiling are printed for every configuration, permanently, beside
    AMI — see the amendment block at FR-013 in 06-requirements.md. The only assertions are
    the two that would indicate a bug rather than a result: that labels help at all, and
    that they do not carry the fitted model ABOVE its own supervised-MLE ceiling, which
    would mean a leak (ADR-0005 revisit trigger).
    """
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    _, _, is_validate, _ = train_label_grid(
        full_scale.emissions, full_scale.codes, state_paths
    )
    _, _, decoded_partial = partial_fit
    oracle = oracle_hmm(full_scale.emissions, full_scale.codes)
    oracle_decoded = np.stack([oracle.viterbi(x) for x in full_scale.emissions.sequences()])

    runs = {
        "unsupervised (T-0003b baseline)": (full_scale.decoded, full_scale.model),
        "label-informed (T-0004b)": (decoded_partial, partial_fit[0]),
        "ORACLE ceiling (supervised MLE)": (oracle_decoded, oracle),
    }
    reports: dict[str, dict[str, object]] = {}
    print("\nAmended FR-013 suite. Scored on the VALIDATE merchant group: never labelled,")
    print("never clamped. Synthetic merchant streams; the generator is in this repo.")
    for name, (decoded, model) in runs.items():
        # Binary PR-AUC gets the smoothed posterior, not the hard decode. The posterior is
        # computed WITHOUT any label clamp, so held-out merchants are scored by a model
        # that has never seen a label of theirs by any route.
        mapping = align_states(
            full_scale.codes[is_validate].ravel(),
            decoded[is_validate].ravel(),
            len(STATE_ORDER),
        )
        healthy_column = int(np.argsort(mapping)[0])
        posterior = np.stack(
            [model.posterior(x)[:, healthy_column] for x in full_scale.emissions.sequences()]
        )
        report = state_recovery_report(
            full_scale.codes[is_validate],
            decoded[is_validate],
            len(STATE_ORDER),
            healthy_code=0,
            non_healthy_score=1.0 - posterior[is_validate],
        )
        lag, flagged, n_bad = detection_lag_windows(
            full_scale.codes[is_validate],
            report["mapping"][decoded[is_validate]] != 0,
        )
        reports[name] = report
        recalls = "  ".join(
            f"{s} {v:.3f}" for s, v in zip(STATE_ORDER, report["recall"], strict=True)
        )
        print(
            f"  {name:34s} ARI {report['ari']:.3f}  AMI {report['ami']:.3f}  "
            f"macro-recall {report['macro_recall']:.3f}  "
            f"binary PR-AUC {report['binary_pr_auc']:.3f} "
            f"(base rate {report['binary_base_rate']:.3f})  "
            f"detection lag {lag:+.1f} windows, flagged {flagged:.2f} of {n_bad}"
        )
        print(f"  {'':34s} per-state recall: {recalls}")

    partial = reports["label-informed (T-0004b)"]
    baseline = reports["unsupervised (T-0003b baseline)"]
    ceiling = reports["ORACLE ceiling (supervised MLE)"]
    assert partial["ami"] > baseline["ami"], (
        "label-informed fitting did not beat unsupervised on AMI; the estimation gap is "
        "not where the literature says it is (ADR-0005 'what would change our mind' #3)"
    )
    assert partial["ari"] <= ceiling["ari"] + 1e-9, (
        f"fitted ARI {partial['ari']:.3f} exceeds the supervised-MLE ceiling "
        f"{ceiling['ari']:.3f} — SUSPECT LABEL LEAKAGE, audit eval/splits.py before "
        "believing any of these numbers (ADR-0005 revisit trigger)"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PRE-REGISTERED PREDICTION THAT FAILED, kept visible on purpose. The K1 literature "
        "survey (project-context/12-lit-survey-k1.md) stated the success bar for the "
        "label-informed HMM in advance as RAMP recall >= 0.35, on the reasoning that closing "
        "the estimation gap should land near the oracle's RAMP recall of 0.343. Measured on "
        "the validate group: unsupervised 0.328, label-informed 0.234 — labels made RAMP "
        "recall WORSE while roughly doubling every other number. The estimation gap closed "
        "for HEALTHY and FRAUD and did not close for the one state the product is named "
        "after. See logbook-entries/T-0004b.md."
    ),
)
def test_ramp_recall_meets_the_surveys_pre_registered_bar(
    full_scale: FullScale, partial_fit: tuple[HMM, np.ndarray, np.ndarray]
) -> None:
    """The survey's own success criterion for item 2, checked rather than rationalised."""
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    _, _, is_validate, _ = train_label_grid(full_scale.emissions, full_scale.codes, state_paths)
    _, _, decoded = partial_fit
    report = state_recovery_report(
        full_scale.codes[is_validate], decoded[is_validate], len(STATE_ORDER), healthy_code=0
    )
    ramp = float(report["recall"][STATE_ORDER.index("RAMP")])
    assert ramp >= 0.35, f"label-informed RAMP recall {ramp:.3f} < 0.35"
