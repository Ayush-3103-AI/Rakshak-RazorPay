"""Rung 7b — the segmented narrative, and the agreement metrics that judge it (T-0124, #58).

``explain/hsmm_onset.py`` (T-0123's explainer half) answers "did this merchant change?" with
**the first structural break in the decoded segmentation** — the first day the Viterbi path
leaves its day-0 state. Its own docstring calls that "a weaker claim than the one Rung 7 will
eventually make", and names this ticket as the one that makes the stronger one. The stronger
claim is *semantic*: not "the path broke here" but "this merchant left HEALTHY here", which is
the quantity ``drift_onset_at`` actually records and the quantity
:func:`~rakshak.eval.metrics.onset_localisation_error` was sealed to score.

Three things are needed to get from one to the other, and all three are here because none of
them needs the HSMM class — they need a decoded path, a mean-dwell vector and a name list.
This package therefore still imports nothing from ``rakshak.models``; see
``hsmm_onset``'s module docstring for why that wall is load-bearing.

**1. Which decoded state is HEALTHY** (:func:`modal_state`). The fit is unsupervised, so the
state indices are arbitrary. HEALTHY is taken to be the state occupying the most decoded
merchant-days across the *fit pool* — a population-level quantity, computed with no reference
label of any kind. That is what keeps the onset estimate an estimate: a HEALTHY chosen per
merchant by looking at its true onset would be scoring itself.

**2. What the other states are called** (:func:`name_states`). By fitted NB mean, relative to
HEALTHY's: the states above it, in ascending order, are the escalation (RAMP then EXFIL); a
state below it is the collapse (BURNT). That is a *rule stated once*, not a fit — it comes
from the generator's own mechanism (``typologies.intensity_multiplier`` raises intensity along
the ramp and then, for ``vanish_after_ramp`` typologies, collapses it), and it is applied
without consulting a single reference label. The alternative — an optimal (Hungarian)
assignment of decoded states to reference states — would maximise the very per-state recall it
is then used to report, which is the goalpost move #58 exists to refuse. If the rule is wrong,
per-state recall collapses and says so.

**3. Agreement, reported the way the survey says to** (:func:`state_agreement`). AMI is the
headline; ARI is printed beside it, never instead of it. Romano, Vinh, Bailey & Verspoor
(JMLR 17, 2016) settle that ARI is the wrong headline when the reference partition is
unbalanced, which this one badly is — most merchant-days are HEALTHY. Both are returned from
one call, in one object, so a caller cannot quote the flattering one alone without deleting
a field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rakshak.explain.registry import ExplanationRequest

__all__ = [
    "Segment",
    "SegmentedTimelineExplainer",
    "StateAgreement",
    "modal_state",
    "name_states",
    "onset_from_healthy",
    "render_timeline",
    "segments",
    "state_agreement",
]


@dataclass(frozen=True, slots=True)
class Segment:
    """One run of a constant state in a decoded path. ``end_day`` is inclusive."""

    state: int
    start_day: int
    end_day: int

    @property
    def length(self) -> int:
        return self.end_day - self.start_day + 1


def segments(path: np.ndarray) -> list[Segment]:
    """Split a Viterbi path into maximal constant-state runs, in day order."""
    states = np.asarray(path, dtype=np.int64)
    if states.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts - 1, [states.size - 1]))
    return [Segment(int(states[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def modal_state(paths: Sequence[np.ndarray], n_states: int) -> int:
    """The state occupying the most decoded days across ``paths`` — the unsupervised HEALTHY.

    Population-level and label-free on purpose: see the module docstring. Ties break to the
    lowest index, which is arbitrary and is the only sane thing to do with a genuine tie.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1; got {n_states}")
    total = np.zeros(n_states, dtype=np.int64)
    for path in paths:
        total += np.bincount(np.asarray(path, dtype=np.int64), minlength=n_states)[:n_states]
    return int(np.argmax(total))


def name_states(nb_mean: np.ndarray, healthy: int, state_names: Sequence[str]) -> tuple[str, ...]:
    """Name each decoded state by its emission mean relative to HEALTHY's.

    ``nb_mean`` is ``(K, C)``; the channel means are summed, which is exact at ``C = 1``
    (the only configuration Rung 7 has ever been run in) and is a defensible total
    otherwise. ``state_names`` is *injected*, as it is for
    :class:`~rakshak.explain.hsmm_onset.HsmmOnsetExplainer`, so this package never imports
    ``models/rung7_hsmm.STATE_NAMES``.

    Convention: ``state_names[0]`` is HEALTHY, ``state_names[-1]`` is the collapsed state,
    and the ones between are the escalation in ascending order of mean. Any state the
    convention has no name for keeps a bare ``"state {i}"``, which is the same honest
    degradation ``HsmmOnsetExplainer`` already falls back to.
    """
    means = np.asarray(nb_mean, dtype=np.float64)
    if means.ndim == 1:
        means = means[:, None]
    totals = means.sum(axis=1)
    k = int(totals.size)
    if not 0 <= healthy < k:
        raise ValueError(f"healthy state {healthy} is not in range(0, {k})")

    names = [f"state {i}" for i in range(k)]
    if state_names:
        names[healthy] = state_names[0]
    ordered = [int(i) for i in np.argsort(totals).tolist() if int(i) != healthy]
    above = [i for i in ordered if totals[i] >= totals[healthy]]
    below = [i for i in ordered if totals[i] < totals[healthy]]
    escalation = list(state_names[1:-1]) if len(state_names) > 2 else []
    collapsed = state_names[-1] if len(state_names) > 1 else ""
    for rank, state in enumerate(above):
        if rank < len(escalation):
            names[state] = escalation[rank]
    if collapsed:
        for state in below:
            names[state] = collapsed
    return tuple(names)


def onset_from_healthy(path: np.ndarray, healthy: int, into: int | None = None) -> float:
    """The day of the first ``HEALTHY -> into`` transition, or ``nan`` if there is none.

    With ``into`` given this is literally the estimator #58 names — the inferred
    ``HEALTHY -> RAMP`` transition. With ``into=None`` it relaxes to ``HEALTHY -> anything``,
    which is the same quantity when the escalation is orderly and a weaker one when it is
    not; the runner reports both, because the gap between them is the model telling you
    whether it found an ordered ramp at all.

    Either way it is a *different* quantity from
    :func:`~rakshak.explain.hsmm_onset.first_change_point`, which takes the first break out
    of whatever state day 0 happened to land in. When the path starts outside HEALTHY the
    two disagree, and this one declines rather than reporting a break that is not an onset.

    ``nan`` is *declining to localise*, which
    :func:`~rakshak.eval.metrics.onset_localisation_error` counts in ``n_unlocalised``
    rather than dropping — a method that fires only on the easy half and reports a
    flattering IQR is the failure that metric was written to expose.
    """
    states = np.asarray(path, dtype=np.int64)
    if states.size < 2:
        return float("nan")
    arrives = states[1:] != healthy if into is None else states[1:] == into
    leaves = np.flatnonzero((states[:-1] == healthy) & arrives)
    return float(leaves[0] + 1) if leaves.size else float("nan")


def render_timeline(
    path: np.ndarray,
    *,
    names: Sequence[str],
    mean_dwell: np.ndarray,
    as_of: int | None = None,
    max_segments: int = 4,
) -> str:
    """The analyst-facing segmented timeline for one merchant.

    "entered RAMP on day 143, 21 days in state, expected dwell 34 days" — the phrasing #58
    asks for. The expected dwell is the *model's* mean dwell for that state, so the analyst
    can see whether a segment is running long or short against what the duration model
    expects, which is the one thing an HSMM knows that a Markov chain does not.

    Only the last ``max_segments`` are rendered. A 300-day path decodes into dozens of runs
    and a wall of them is not an explanation.
    """
    found = segments(path)
    if not found:
        return "No decoded history for this merchant, so there is no timeline to show."
    end = int(as_of if as_of is not None else found[-1].end_day)
    dwell = np.asarray(mean_dwell, dtype=np.float64)

    def label(state: int) -> str:
        return names[state] if state < len(names) else f"state {state}"

    lines = []
    for seg in found[-max_segments:]:
        held = min(seg.end_day, end) - seg.start_day + 1
        expected = float(dwell[seg.state]) if seg.state < dwell.size else float("nan")
        lines.append(
            f"entered {label(seg.state)} on day {seg.start_day}, {held} day(s) in state, "
            f"expected dwell {expected:.0f} days"
        )
    head = f"Segmented timeline through day {end} ({len(found)} regime(s) decoded in all):"
    return head + "\n  - " + "\n  - ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Agreement: AMI as the headline, ARI beside it
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StateAgreement:
    """How well a decoded segmentation recovers the reference states.

    ``ami`` is the headline and ``ari`` sits beside it, both always populated: the two
    fields exist together so that quoting only the flattering one requires deleting a
    field rather than choosing a function. Romano et al. (JMLR 17, 2016) is the reason —
    ARI's null model assumes balanced clusters, and a merchant-day partition that is mostly
    HEALTHY is nowhere near balanced.

    ``recall`` is per reference state under the *unsupervised* naming rule
    (:func:`name_states`), never an optimal assignment. ``macro_recall`` is its unweighted
    mean, which is the number that refuses to be carried by the majority state.
    """

    ami: float
    ari: float
    recall: dict[str, float]
    macro_recall: float
    support: dict[str, int]
    n: int


def state_agreement(predicted: np.ndarray, reference: np.ndarray) -> StateAgreement:
    """AMI (headline), ARI (beside it), per-state recall and its macro-average.

    Both arrays are one entry per merchant-day and carry **state names**, not indices —
    the caller has already applied :func:`name_states`, so recall is measured against the
    naming rule that was declared rather than against one chosen to flatter it.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    pred = np.asarray(predicted).astype(str)
    ref = np.asarray(reference).astype(str)
    if pred.shape != ref.shape:
        raise ValueError("predicted and reference must be the same length")
    if pred.size == 0:
        return StateAgreement(
            ami=float("nan"),
            ari=float("nan"),
            recall={},
            macro_recall=float("nan"),
            support={},
            n=0,
        )

    recall: dict[str, float] = {}
    support: dict[str, int] = {}
    for name in sorted(set(ref.tolist())):
        member = ref == name
        support[name] = int(member.sum())
        recall[name] = float((pred[member] == name).mean())
    return StateAgreement(
        ami=float(adjusted_mutual_info_score(ref, pred)),
        ari=float(adjusted_rand_score(ref, pred)),
        recall=recall,
        macro_recall=float(np.mean(list(recall.values()))),
        support=support,
        n=int(pred.size),
    )


# ─────────────────────────────────────────────────────────────────────────────
# The registered explainer
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class SegmentedTimelineExplainer:
    """Stage-2 narrative: the decoded phase history, named, with expected dwells.

    Registers **beside** ``hsmm_onset``, not instead of it, and beside the ``pred_contrib``
    reason codes the scoring rung already emits — #58 is explicit that the timeline is
    additional. It has no ``predict``, so ``explain.registry.register`` accepts it and the
    scoring path cannot reach it; that refusal is the whole point of the register.

    ``paths`` maps merchant id to its decoded Viterbi path over day 0..T. Decoding once at
    measurement time and holding the result keeps Stage 2 a lookup, which is what the 50 ms
    budget needs — and that budget is still **not certified**; see ``rung7_hsmm``'s
    docstring.
    """

    paths: dict[str, np.ndarray]
    names: tuple[str, ...]
    mean_dwell: np.ndarray
    healthy: int = 0

    @property
    def name(self) -> str:
        return "hsmm_segmented_timeline"

    def explain(self, request: ExplanationRequest) -> str:
        path = self.paths.get(request.merchant_id)
        if path is None or request.day < 0:
            return (
                f"No decoded state path is loaded for {request.merchant_id}, so this "
                f"explainer cannot show its phase history. The {request.action.name} "
                f"decision stands on the score alone."
            )
        window = np.asarray(path)[: request.day + 1]
        onset = onset_from_healthy(window, self.healthy)
        healthy_name = self.names[self.healthy] if self.healthy < len(self.names) else "HEALTHY"
        opening = (
            f"{request.merchant_id} has not left {healthy_name} in the decoded history up "
            f"to day {request.day}."
            if np.isnan(onset)
            else (
                f"{request.merchant_id} left {healthy_name} on day {int(onset)}, "
                f"{request.day - int(onset)} day(s) before this {request.action.name}."
            )
        )
        return (
            opening
            + "\n"
            + render_timeline(
                window, names=self.names, mean_dwell=self.mean_dwell, as_of=request.day
            )
        )
