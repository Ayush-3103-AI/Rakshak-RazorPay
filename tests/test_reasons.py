"""T-0013 / FR-014 — the reason string, and the two properties that make it honest.

FR-014's `Verified by` clause is *"golden-file test on fixed fixtures"*, and
`test_golden_reason_string` is it: an HMM whose parameters are written down in this
file, a sequence written down in this file, and the exact sentence pinned character
for character. It needs no parquet and no fit, so it runs on a fresh clone and it
fails loudly the moment anyone edits the phrasing — which is the point, because the
phrasing is what a merchant hears.

Two properties matter more than the string itself.

`test_contributions_sum_to_the_log_ratio` pins the claim that the top-3 emissions
are a **decomposition and not an attribution**. The emission covariance is diagonal,
so the per-feature terms sum to the exact log-likelihood ratio between the two
states with no residual. If anyone swaps in a full covariance, this test fails
rather than the repo quietly starting to publish made-up percentages.

`test_explanation_ignores_the_future` is the temporal guarantee, and it is the same
shape as `test_hmm_score.py::test_filtered_posterior_ignores_the_future`: overwrite
every window *after* the flag with garbage and require the whole `Reason` to come
back unchanged. `test_the_truncation_test_has_teeth` is its negative control — it
runs the identical mutation against an untruncated Viterbi decode and requires the
path to move, so the guarantee cannot be passing vacuously.

Numbers touched here are measured on SYNTHETIC merchant streams with injected
typologies; the generator is in this repo (CLAUDE.md non-negotiable #3).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rakshak.config import (
    N_HIDDEN_STATES,
    SEED,
    STATE_PATHS_PARQUET,
    TRANSACTIONS_PARQUET,
)
from rakshak.eval.splits import load_split
from rakshak.explain import reasons as R
from rakshak.models.hmm import HMM

needs_data = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator --seed 42` first",
)

FIXTURE_FEATURES: tuple[str, ...] = ("log_velocity", "new_payer_ratio", "refund_ratio")
"""Three real feature names, so the golden string exercises `FEATURE_PHRASES`
rather than the fall-through that renders an unknown feature under its own name."""


def _fixture_model() -> HMM:
    """An HMM with hand-written parameters. No fitting, no data, no seed.

    State order follows `hmm_score.STATE_ORDER` — HEALTHY, RAMP, FRAUD, DORMANT —
    because label clamping pins that identity in the shipping model and the
    explain layer reads `BAD_COLUMNS` off it.
    """
    model = HMM(n_states=N_HIDDEN_STATES, n_features=len(FIXTURE_FEATURES))
    model.mu = np.array(
        [
            [0.0, 0.0, 0.0],  # HEALTHY — at the merchant's own baseline
            [2.0, 2.0, 0.0],  # RAMP    — volume and new payers up together
            [4.0, 1.0, 3.0],  # FRAUD   — volume far up, refunds up
            [-3.0, 0.0, 0.0],  # DORMANT — volume collapsed
        ]
    )
    model.var = np.ones((N_HIDDEN_STATES, len(FIXTURE_FEATURES)))
    model.log_pi = np.log([0.97, 0.01, 0.01, 0.01])
    model.log_A = np.log(
        [
            [0.90, 0.07, 0.02, 0.01],
            [0.05, 0.80, 0.10, 0.05],
            [0.02, 0.05, 0.85, 0.08],
            [0.02, 0.02, 0.06, 0.90],
        ]
    )
    return model


def _fixture_sequence() -> np.ndarray:
    """Four baseline windows, then four windows sitting on RAMP's mean."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            # Deliberately NOT sitting exactly on RAMP's mean: equal contributions
            # would make the reported feature order a tie-break rather than a
            # result, and the ordering assertion below would prove nothing.
            [2.6, 1.6, 0.0],
            [2.5, 1.7, 0.0],
            [2.7, 1.5, 0.0],
            [2.6, 1.6, 0.0],
        ]
    )


def _fixture_case() -> tuple[HMM, np.ndarray, np.ndarray, np.ndarray]:
    """`(model, sequence, eligible, window_start_day)` for the golden fixture."""
    sequence = _fixture_sequence()
    eligible = np.ones(len(sequence), dtype=bool)
    window_start_day = np.arange(len(sequence)) * 7
    return _fixture_model(), sequence, eligible, window_start_day


def _explain_fixture() -> R.Reason:
    model, sequence, eligible, days = _fixture_case()
    reason = R.explain_merchant(
        model, "M-FIXTURE", sequence, eligible, days, FIXTURE_FEATURES
    )
    assert reason is not None, "the fixture must flag, or it tests nothing"
    return reason


# ---------------------------------------------------------------------------
# The decomposition is exact
# ---------------------------------------------------------------------------


def test_per_feature_log_emission_sums_to_log_emission() -> None:
    """`log_emission_per_feature` summed over D reproduces `HMM.log_emission`."""
    model = _fixture_model()
    x = np.array([1.3, -0.4, 2.2])
    np.testing.assert_allclose(
        R.log_emission_per_feature(model, x).sum(axis=1),
        model.log_emission(x[None, :])[0],
        rtol=0,
        atol=1e-12,
    )


def test_contributions_sum_to_the_log_ratio() -> None:
    """The reported contributions are a decomposition, with no residual.

    This is what separates the reason string from a plausible-sounding
    attribution: the D per-feature numbers add up to the exact quantity the model
    used to prefer one state over the other, because the covariance is diagonal.
    """
    model = _fixture_model()
    x = np.array([2.4, 1.7, 0.3])
    contributions = R.emission_contributions(model, x, new_state=1, previous_state=0)
    log_b = model.log_emission(x[None, :])[0]
    np.testing.assert_allclose(
        contributions.sum(), log_b[1] - log_b[0], rtol=0, atol=1e-12
    )


# ---------------------------------------------------------------------------
# The golden file (FR-014's `Verified by`)
# ---------------------------------------------------------------------------


GOLDEN_TEXT: str = (
    "On 4 February 2026 this account moved into a pattern we flag as "
    "rapid-volume-escalation. The measurements that moved it there, each compared "
    "against this account's own trading history rather than against other merchants: "
    "transaction volume ran well above this account's own baseline (+2.60 SD); share "
    "of first-time payers ran well above this account's own baseline (+1.60 SD). What "
    "would resolve this: recent invoices or purchase orders covering the increase, "
    "and the campaign or sales channel that drove it."
)
"""Pinned character for character. Editing the phrasing must break this test —
the sentence is a deliverable, not an implementation detail."""


def test_golden_reason_string() -> None:
    """FR-014: the rendered string, on a fixture with no data dependency."""
    assert _explain_fixture().text == GOLDEN_TEXT


def test_golden_reason_names_state_date_and_top_emissions() -> None:
    """FR-014's three required elements are present as structured data too.

    The string is for the merchant; these fields are for T-0014's viewer, which
    must render them rather than re-deriving anything from prose.
    """
    reason = _explain_fixture()
    assert reason.state == "RAMP"  # (a) the state transitioned into
    assert reason.transition_date == "4 February 2026"  # (b) the date
    assert reason.transition_day == 34  # window 4 starts day 28, ends day 34
    # (c) top emissions, ordered by contribution, never more than three
    assert len(reason.top_features) <= R.TOP_N_FEATURES
    # Ordered by contribution on merit, not by a tie-break: `log_velocity` sits
    # 2.60 SD out against `new_payer_ratio`'s 1.60 SD.
    assert [f.feature for f in reason.top_features] == [
        "log_velocity",
        "new_payer_ratio",
    ]
    contributions = [f.contribution for f in reason.top_features]
    assert contributions == sorted(contributions, reverse=True)
    assert all(f.contribution > 0 for f in reason.top_features)


def test_refund_ratio_is_excluded_because_it_argues_against_the_transition() -> None:
    """Only positive evidence is reported, and this fixture proves the filter bites.

    `refund_ratio` sits at 0.0, which is HEALTHY's mean and RAMP's mean alike, so
    it carries no evidence either way and must not pad the list to three.
    """
    assert "refund_ratio" not in {f.feature for f in _explain_fixture().top_features}


# ---------------------------------------------------------------------------
# The temporal guarantee
# ---------------------------------------------------------------------------


def test_explanation_ignores_the_future() -> None:
    """Overwriting every window after the flag cannot change the explanation."""
    model, sequence, eligible, days = _fixture_case()
    baseline = R.explain_merchant(
        model, "M-FIXTURE", sequence, eligible, days, FIXTURE_FEATURES
    )
    assert baseline is not None

    mutated = sequence.copy()
    mutated[baseline_index(baseline, days) + 1 :] = np.array([-9.0, -9.0, 9.0])
    after = R.explain_merchant(
        model, "M-FIXTURE", mutated, eligible, days, FIXTURE_FEATURES
    )
    assert after == baseline


def baseline_index(reason: R.Reason, days: np.ndarray) -> int:
    """Window index the flag fired on, recovered from its window-END day."""
    from rakshak.config import WINDOW_DAYS

    return int(np.flatnonzero(days == reason.flag_day - WINDOW_DAYS + 1)[0])


def test_the_truncation_test_has_teeth() -> None:
    """Negative control: an untruncated decode DOES move when the future changes.

    Without this, `test_explanation_ignores_the_future` could be passing because
    the fixture is insensitive rather than because the decode is truncated.
    """
    model, sequence, _, _ = _fixture_case()
    mutated = sequence.copy()
    mutated[5:] = np.array([-9.0, -9.0, 9.0])
    assert not np.array_equal(model.viterbi(sequence), model.viterbi(mutated)), (
        "the fixture is insensitive to its own tail; the truncation test above is vacuous"
    )


def test_unflagged_merchant_returns_none() -> None:
    """A merchant whose belief never crosses the threshold gets no reason."""
    model, _, eligible, days = _fixture_case()
    quiet = np.zeros((8, len(FIXTURE_FEATURES)))
    assert R.explain_merchant(model, "M-QUIET", quiet, eligible, days, FIXTURE_FEATURES) is None


def test_ineligible_windows_cannot_raise_a_flag() -> None:
    """The decision-window mask gates the flag, so it gates the explanation."""
    model, sequence, _, days = _fixture_case()
    eligible = np.zeros(len(sequence), dtype=bool)
    eligible[:4] = True  # only the four baseline windows are in the decision window
    assert (
        R.explain_merchant(model, "M-MASKED", sequence, eligible, days, FIXTURE_FEATURES)
        is None
    )


# ---------------------------------------------------------------------------
# Invariants that hold for every merchant, not just the fixture
# ---------------------------------------------------------------------------


def test_previous_state_is_always_healthy_on_the_fixture() -> None:
    """The walk-back stops at the departure from HEALTHY, by construction.

    HEALTHY is the only state outside `BAD_COLUMNS`, so a walk-back that consumes
    bad states can only stop on HEALTHY or on the start of the sequence. This is
    what stops the string explaining a RAMP -> FRAUD step as though it were the
    reason the merchant was flagged at all.
    """
    assert _explain_fixture().previous_state == "HEALTHY"


def test_disagreement_branch_says_so_rather_than_inventing_a_transition() -> None:
    """When the MAP path stays HEALTHY the string must not claim a transition.

    Driven through `_render` directly: constructing emissions that split belief
    across three bad states without tipping the MAP path is fiddly and would make
    the test about the fixture rather than about the wording.
    """
    text = R._render(
        {
            "transition_date": "1 March 2026",
            "flag_date": "1 March 2026",
            "pattern": R.PATTERN_LABELS["RAMP"],
            "state": "RAMP",
            "state_at_flag": "RAMP",
            "viterbi_agrees": False,
        },
        (),
    )
    assert "passed the review threshold" in text
    assert "moved into a pattern" not in text
    assert "No single measurement drove this on its own" in text


def test_escalation_between_transition_and_flag_is_stated() -> None:
    """A merchant that left HEALTHY as RAMP but was flagged as FRAUD is told both."""
    text = R._render(
        {
            "transition_date": "1 March 2026",
            "flag_date": "15 March 2026",
            "pattern": R.PATTERN_LABELS["RAMP"],
            "state": "RAMP",
            "state_at_flag": "FRAUD",
            "viterbi_agrees": True,
        },
        (),
    )
    assert "On 1 March 2026 this account moved into" in text
    assert "By 15 March 2026 the pattern had changed to" in text
    # Remediation follows the CURRENT state, not the state it entered two weeks ago.
    assert R.REMEDIATION["FRAUD"] in text


def test_every_bad_state_has_a_label_and_a_remediation() -> None:
    """A new latent state must not render as a KeyError in front of a merchant."""
    from rakshak.models.hmm_score import BAD_COLUMNS, STATE_ORDER

    for column in BAD_COLUMNS:
        assert STATE_ORDER[column] in R.PATTERN_LABELS
        assert STATE_ORDER[column] in R.REMEDIATION


def test_clipped_levels_are_not_reported_as_measured() -> None:
    """A level at `Z_CLIP` is a floor, and the string has to say so."""
    from rakshak.config import Z_CLIP

    text = R._render(
        {
            "transition_date": "1 March 2026",
            "flag_date": "1 March 2026",
            "pattern": R.PATTERN_LABELS["FRAUD"],
            "state": "FRAUD",
            "state_at_flag": "FRAUD",
            "viterbi_agrees": True,
        },
        (
            R.FeatureContribution(
                feature="refund_ratio",
                phrase=R.FEATURE_PHRASES["refund_ratio"],
                z=Z_CLIP,
                contribution=12.0,
            ),
        ),
    )
    assert "beyond the range we can measure" in text
    assert "where the scale is capped" in text


def test_calendar_dates_are_locale_independent() -> None:
    """Month names come from this repo, not from the host's locale (NFR-003)."""
    assert R._calendar_date(0) == "1 January 2026"
    assert R._calendar_date(240) == "29 August 2026"


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------


@needs_data
def test_reasons_json_is_byte_identical_for_a_fixed_seed() -> None:
    """T-0013's `Done when`: same seed -> byte-identical `results/reasons.json`."""
    split = load_split(R.REASONS_SPLIT, unlock_test=R.UNLOCK_TICKET)
    first = R.render_json(R.build_reasons(SEED, split=split), SEED, split)
    second = R.render_json(R.build_reasons(SEED, split=split), SEED, split)
    assert first == second


@needs_data
def test_reasons_json_carries_provenance_and_honest_counts() -> None:
    """Every artifact in this repo states which script, seed and split made it."""
    split = load_split(R.REASONS_SPLIT, unlock_test=R.UNLOCK_TICKET)
    document = json.loads(R.render_json(R.build_reasons(SEED, split=split), SEED, split))

    provenance = document["provenance"]
    assert provenance["seed"] == SEED
    assert provenance["split"] == "test"
    assert provenance["contribution_units"] == "nats"
    assert "DECODE, not a detection" in provenance["transition_day_note"]

    counts = document["counts"]
    # The false positives are counted and published, not filtered out of the file.
    assert counts["flagged_and_healthy"] > 0, (
        "at this model's measured precision the file must contain false positives; "
        "a reasons file holding only true positives would be selection, not evidence"
    )
    assert (
        counts["flagged_and_truly_bad"] + counts["flagged_and_healthy"]
        == counts["merchants_flagged"]
        == len(document["reasons"])
    )


@needs_data
def test_every_reason_explains_a_departure_from_healthy() -> None:
    """The walk-back invariant, on the real split rather than the fixture."""
    split = load_split(R.REASONS_SPLIT, unlock_test=R.UNLOCK_TICKET)
    built = R.build_reasons(SEED, split=split)
    assert built, "the test split must flag someone, or this proves nothing"
    for reason in built:
        assert reason.previous_state == "HEALTHY"
        assert reason.state in R.PATTERN_LABELS
        assert reason.transition_day <= reason.flag_day
        assert reason.text.endswith(".")
        assert len(reason.top_features) <= R.TOP_N_FEATURES
