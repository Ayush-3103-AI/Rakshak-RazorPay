"""Rung 7 as an **explainer**: the registered explainer, and the onset estimator it uses.

``models/rung7_hsmm.py`` (T-0123, #57) is an inference core. It had never been run on real
data and never scored against ``onset_localisation_error``, the metric
``docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md`` §2 declared and ``EVAL-LOCK-CYCLE3.json``
sealed for it. Rung 7's presence in the pipeline is two halves, and they live on opposite
sides of a wall:

1. **This module** — :class:`HsmmOnsetExplainer`, the registered explainer. It satisfies
   ``explain.registry.Explainer`` and, critically, **does not** satisfy ``Scorer``: it has
   no ``predict``, so ``register`` accepts it and the scoring path cannot reach it.
2. ``rakshak.score_rung7`` — the runner. It fits a pooled HSMM on TRAIN-fold sequences,
   decodes a change-point per VAL-fold merchant with a known onset, and writes the
   explanation-quality artifact.

**Why the runner is not in this file, where it started.**
``tests/unit/test_explain_registry.py::test_the_explain_package_does_not_import_the_models_package``
refuses any import of ``rakshak.models`` from anywhere under ``explain/``, and its sibling
refuses the reverse. Between them they are what stops an explainer acquiring a scoring path
by accident, and that is worth more than the convenience of one file. The runner genuinely
needs ``rung7_hsmm.fit`` — it fits the model — so the runner moved out and the wall stayed.

What is left here needs a **decoder**, not the HSMM class: both :func:`first_change_point`
and the explainer call ``.decode(obs)`` and nothing else. So they take one structurally, as
:class:`Decoder`, and this package imports nothing from ``models/``. That is the same
reason ``explain.registry.Scorer`` is a Protocol rather than a base class — the wall does
not need a dependency to stand.

**Why Rung 7 is not a ladder row, structurally and not by convention.**
``artifacts/build.py::read_result_rows`` globs ``data/v2/eval/*.json`` and turns every file
it finds into a row of ``ladder.json``. A Rung 7 file in that directory therefore *becomes*
a ladder row no matter what it says inside it — it would sit in the results table beside
Rungs 0-6 with a blank PR-AUC column, and a reader would reasonably conclude Rung 7 scored
badly rather than that it was never a scorer. So the artifact goes to
``score_rung7.EXPLANATION_DIR`` instead. #51 is explicit that Rung 7 runs "at Stage 2 of the
cascade only — on non-PASS decisions, never in the scoring path, never scored on PR-AUC",
and the directory is what enforces it.

**What the estimator actually is, stated plainly.** The fit is unsupervised and univariate.
Nothing tells it which of the K decoded states is "healthy" in ``rung7_hsmm.STATE_NAMES``
terms, so a semantic estimator ("the first day in the EXFIL state") is not available without
the narrative layer T-0124 owns. The estimator here is **the first structural break in the
decoded segmentation** — the first day the Viterbi path leaves its day-0 state. That is what
"when did drift begin" reduces to for an unlabelled onset, and it is a weaker claim than the
one Rung 7 will eventually make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rakshak.explain.registry import ExplanationRequest

__all__ = [
    "Decoder",
    "HsmmOnsetExplainer",
    "first_change_point",
]


class Decoder(Protocol):
    """Whatever can turn an observation sequence into a state path.

    Structural on purpose: ``rung7_hsmm.HsmmNb`` satisfies it, and so does any test double,
    without this package importing either. See the module docstring for why that matters —
    naming ``HsmmNb`` here is the one thing that would put ``explain/`` back on the wrong
    side of the wall, for a type annotation nobody needs at run time.
    """

    def decode(self, obs: np.ndarray) -> np.ndarray: ...


def first_change_point(model: Decoder, obs: np.ndarray) -> float:
    """The day the Viterbi path first leaves its day-0 state, or ``nan`` if it never does.

    ``nan`` is *declining to localise*, which
    :func:`~rakshak.eval.metrics.onset_localisation_error` counts as ``n_unlocalised``
    rather than dropping. That distinction is the reason this returns ``nan`` instead of
    falling back to a guess: a method that only fires on the easy half and reports a
    flattering IQR is the failure the metric was written to expose.
    """
    path = model.decode(obs)
    boundaries = np.flatnonzero(np.diff(path))
    return float(boundaries[0] + 1) if boundaries.size else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# The registered explainer
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class HsmmOnsetExplainer:
    """Stage-2 narrative for one non-PASS merchant-day: *when* this merchant changed.

    **Deliberately has no ``predict``.** ``explain.registry.register`` refuses anything
    satisfying ``Scorer``, and that refusal is the whole point of the register: an HSMM
    fitted on, and decoded over, the subset another rung already promoted would produce a
    PR-AUC computed on a population selected by a different model. This class is what the
    register was built to accept, and the fitted HSMM — which it holds rather than inherits
    from — is what the register was built to keep out.

    ``sequences`` maps merchant id to its daily count vector.
    :class:`~rakshak.explain.registry.ExplanationRequest` carries the decision, not the
    merchant's history, so the history is held here. A merchant with no sequence gets an
    honest "cannot say" rather than a fabricated day: at Stage 2 the alternative to an
    explanation is silence, never a guess.

    ``state_names`` is **injected rather than imported** — it belongs to
    ``models/rung7_hsmm.py`` and importing it here would breach the wall for one narrative
    noun. ``rakshak.score_rung7`` passes the real tuple. Left empty, the narrative falls
    back to a bare state index, which is the same degradation the bounds check below always
    allowed for an out-of-range state.
    """

    model: Decoder
    sequences: dict[str, np.ndarray] = field(default_factory=dict)
    state_names: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return "hsmm_onset"

    def explain(self, request: ExplanationRequest) -> str:
        obs = self.sequences.get(request.merchant_id)
        if obs is None or request.day < 1:
            return (
                f"No daily-count history is loaded for {request.merchant_id}, so this "
                f"explainer cannot say when its behaviour changed. The "
                f"{request.action.name} decision stands on the score alone."
            )
        path = self.model.decode(np.asarray(obs)[: request.day + 1])
        breaks = np.flatnonzero(np.diff(path))
        if not breaks.size:
            return (
                f"{request.merchant_id} shows one continuous behavioural regime through "
                f"day {request.day}: the duration model finds no change-point, so the "
                f"{request.action.name} rests on level, not on a change."
            )
        onset = int(breaks[-1]) + 1
        last = int(path[-1])
        state = self.state_names[last] if last < len(self.state_names) else f"state {last}"
        return (
            f"{request.merchant_id} entered its current transaction-volume regime "
            f"({state}) on day {onset}, {request.day - onset} day(s) before this "
            f"{request.action.name}. The regime before it had held for "
            f"{onset - (int(breaks[-2]) + 1) if breaks.size > 1 else onset} day(s)."
        )
