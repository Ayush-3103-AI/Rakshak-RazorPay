"""T-0124 — Rung 7b's segmentation, naming, onset estimator and agreement reporting.

The runner (``scripts/rung7b_score.py``) needs a fitted HSMM and a 2.7 GB dataset. Every
decision it makes about *what a decoded path means*, though, is in
``rakshak.explain.segmentation`` and is testable on a hand-written path — so that is what
this file tests, on paths whose right answer is arithmetic rather than a model output.

The two assertions that carry the ticket's weight:

* :func:`test_onset_is_the_first_departure_from_healthy_not_the_first_break` — the whole
  difference between 7a's estimator and 7b's. A path that starts *outside* HEALTHY has a
  structural break that is not an onset, and 7b must not report it as one.
* :func:`test_ami_and_ari_disagree_on_an_unbalanced_reference` — the K1 survey's finding,
  reproduced on this project's own partition shape rather than cited. It is the reason
  ``StateAgreement`` carries both numbers in one object.
"""

from __future__ import annotations

import numpy as np
import pytest

from rakshak.explain.registry import ExplanationRequest, Scorer
from rakshak.explain.segmentation import (
    SegmentedTimelineExplainer,
    modal_state,
    name_states,
    onset_from_healthy,
    render_timeline,
    segments,
    state_agreement,
)
from rakshak.schemas import Action

STATE_NAMES = ("HEALTHY", "RAMP", "EXFIL", "BURNT")


def test_segments_are_maximal_runs_covering_the_whole_path() -> None:
    path = np.array([0, 0, 0, 1, 1, 2])
    found = segments(path)
    assert [(s.state, s.start_day, s.end_day, s.length) for s in found] == [
        (0, 0, 2, 3),
        (1, 3, 4, 2),
        (2, 5, 5, 1),
    ]
    assert sum(s.length for s in found) == path.size
    assert segments(np.array([], dtype=np.int64)) == []


def test_modal_state_is_the_most_occupied_state_across_paths() -> None:
    paths = [np.array([2, 2, 2, 1]), np.array([2, 0, 0])]
    assert modal_state(paths, n_states=3) == 2


def test_naming_puts_the_escalation_above_healthy_and_the_collapse_below() -> None:
    # state 1 is the quiet one, state 0 is HEALTHY, 2 and 3 escalate.
    nb_mean = np.array([[10.0], [2.0], [25.0], [14.0]])
    assert name_states(nb_mean, healthy=0, state_names=STATE_NAMES) == (
        "HEALTHY",
        "BURNT",
        "EXFIL",
        "RAMP",
    )


def test_naming_leaves_a_state_the_rule_has_no_name_for_generic() -> None:
    """Three states above HEALTHY and none below: BURNT is not claimed by anyone.

    This is a real outcome of the unsupervised rule, not a bug to paper over. The
    alternative — handing the spare name to whichever state maximises agreement — is the
    optimal assignment #58 refuses, so the honest degradation is a bare index and a BURNT
    recall of zero.
    """
    nb_mean = np.array([[5.0], [6.0], [7.0], [8.0]])
    assert name_states(nb_mean, healthy=0, state_names=STATE_NAMES) == (
        "HEALTHY",
        "RAMP",
        "EXFIL",
        "state 3",
    )


def test_onset_is_the_first_departure_from_healthy_not_the_first_break() -> None:
    healthy = 0
    # Starts in state 2, breaks at day 2, but never leaves HEALTHY until day 5.
    path = np.array([2, 2, 0, 0, 0, 1, 1])
    assert onset_from_healthy(path, healthy) == 5.0


def test_onset_declines_rather_than_guessing() -> None:
    assert np.isnan(onset_from_healthy(np.array([0, 0, 0, 0]), healthy=0))
    assert np.isnan(onset_from_healthy(np.array([1, 1, 2, 2]), healthy=0))
    assert np.isnan(onset_from_healthy(np.array([0]), healthy=0))


def test_timeline_names_the_state_the_day_and_the_expected_dwell() -> None:
    path = np.concatenate([np.zeros(140, dtype=int), np.ones(21, dtype=int)])
    text = render_timeline(
        path, names=STATE_NAMES, mean_dwell=np.array([88.0, 34.0, 12.0, 5.0]), as_of=160
    )
    assert "entered RAMP on day 140, 21 day(s) in state, expected dwell 34 days" in text
    assert "entered HEALTHY on day 0" in text


def test_agreement_reports_ami_ari_and_per_state_recall_together() -> None:
    ref = np.array(["HEALTHY"] * 6 + ["RAMP"] * 2)
    pred = np.array(["HEALTHY"] * 5 + ["RAMP"] + ["RAMP"] + ["HEALTHY"])
    result = state_agreement(pred, ref)
    assert result.n == 8
    assert result.support == {"HEALTHY": 6, "RAMP": 2}
    assert result.recall["HEALTHY"] == pytest.approx(5 / 6)
    assert result.recall["RAMP"] == pytest.approx(0.5)
    assert result.macro_recall == pytest.approx((5 / 6 + 0.5) / 2)
    assert np.isfinite(result.ami) and np.isfinite(result.ari)


def test_ami_and_ari_disagree_on_an_unbalanced_reference() -> None:
    """ARI reads higher than AMI on a ~90/6/3/2 partition with the same errors.

    Romano, Vinh, Bailey & Verspoor (JMLR 17, 2016) is why #58 fixes AMI as the headline:
    ARI's null model assumes balanced clusters, so on a partition dominated by one state
    it flatters a clustering that mostly gets the dominant state right. Asserted here on
    this project's own partition shape so the claim is measured, not cited.
    """
    rng = np.random.default_rng(7)
    ref = np.array(["HEALTHY"] * 900 + ["RAMP"] * 60 + ["EXFIL"] * 30 + ["BURNT"] * 20)
    pred = ref.copy()
    # Scramble the minority states badly, the majority state hardly at all.
    minority = np.flatnonzero(ref != "HEALTHY")
    pred[rng.choice(minority, size=int(0.7 * minority.size), replace=False)] = "HEALTHY"
    result = state_agreement(pred, ref)
    assert result.ari > result.ami
    assert result.recall["HEALTHY"] > result.macro_recall


def test_the_timeline_explainer_cannot_reach_the_scoring_path() -> None:
    explainer = SegmentedTimelineExplainer(
        paths={"M000001": np.array([0] * 10 + [1] * 5)},
        names=STATE_NAMES,
        mean_dwell=np.array([50.0, 20.0, 10.0, 5.0]),
        healthy=0,
    )
    assert not isinstance(explainer, Scorer)
    request = ExplanationRequest(
        merchant_id="M000001",
        day=14,
        x=np.zeros(0),
        columns=(),
        score=float("nan"),
        action=Action.HOLD,
    )
    text = explainer.explain(request)
    assert "left HEALTHY on day 10" in text
    assert "entered RAMP on day 10" in text
    missing = explainer.explain(
        ExplanationRequest(
            merchant_id="nobody",
            day=14,
            x=np.zeros(0),
            columns=(),
            score=float("nan"),
            action=Action.HOLD,
        )
    )
    assert "cannot show its phase history" in missing
