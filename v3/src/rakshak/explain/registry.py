"""The explainer register — and the wall between explaining and scoring (T-0118).

An **explainer** says *why* a merchant-day looks the way it does. A **scorer** says how
risky it is, and its number goes into the cost layer, the capacity selector and every
committed metric. They are different roles and this module exists to keep them different.

The reason is specific and not stylistic. Rung 7 is an HSMM that runs at Stage 2 only, on
the merchants a cheaper rung already promoted. Its state posterior is a superb *narrative*
— "this merchant moved into a high-refund regime on day 143" — and a terrible headline
score, because it has only ever seen the promoted subset. If it were reachable as a scoring
rung, someone would eventually score it, and the resulting PR-AUC would be computed on a
population selected by another model. That number would look ordinary and be meaningless.

So registration **refuses** anything that satisfies :class:`Scorer`. The bar is structural,
checked at registration time, and cannot be satisfied by remembering not to do it.

The registration surface lands here with no explainer registered against it. That is
deliberate: docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md §4 commits to landing the eval-side
surface with no rung attached, so ``EVAL-LOCK-CYCLE3.json`` can hash it first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from rakshak.schemas import Action

__all__ = [
    "EXPLAINERS",
    "ExplanationRequest",
    "Explainer",
    "NotAnExplainerError",
    "Scorer",
    "get",
    "register",
    "registered",
    "reset_for_testing",
]


@runtime_checkable
class Scorer(Protocol):
    """A scoring rung: emits the calibrated probability the cost layer consumes.

    Structural, matching what ``cli.py`` actually calls on a trained rung
    (``model.predict(rows.x, rows.columns)``). Declared here rather than in
    ``rakshak.models`` on purpose — this module is the only place that needs to *exclude*
    it, and ``src/rakshak/models/`` is a scoring package that should not have to import
    the explain package to define its own contract. Nothing imports the other way either;
    ``tests/unit/test_explain_registry.py`` asserts that.
    """

    def predict(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ExplanationRequest:
    """One merchant-day, and the decision already taken on it.

    An explainer receives the ``score`` and ``action`` as *given*. It cannot change them,
    which is what makes an explanation an explanation rather than a second opinion that
    quietly becomes the decision. A frozen request object rather than positional arguments
    so a later explainer can be handed more context without every implementation breaking.
    """

    merchant_id: str
    day: int
    x: np.ndarray
    columns: tuple[str, ...]
    score: float
    action: Action


@runtime_checkable
class Explainer(Protocol):
    """Merchant-readable reason for one merchant-day. Deliberately has no ``predict``.

    ``explain`` returns a string an ops analyst can read to a merchant on the phone. It is
    not required to be short and it is required to be true: an explanation that describes
    a mechanism the decision did not actually use is worse than no explanation, because it
    survives the phone call and then fails an audit.
    """

    @property
    def name(self) -> str: ...

    def explain(self, request: ExplanationRequest) -> str: ...


class NotAnExplainerError(TypeError):
    """Raised when something that can score is offered as an explainer."""


#: name -> the single shared instance. Empty at T-0118, on purpose.
EXPLAINERS: dict[str, Explainer] = {}


def register(explainer: Explainer) -> Explainer:
    """Add an explainer to the register. Refuses anything that is also a :class:`Scorer`.

    Returns the explainer, so it can be used as a decorator on an instance-producing
    expression or simply called for its side effect.
    """
    if isinstance(explainer, Scorer):
        raise NotAnExplainerError(
            f"{type(explainer).__name__} defines predict(), so it satisfies Scorer and "
            "cannot register as an explainer. An explainer runs at Stage 2 on merchants "
            "another rung already promoted; scoring it would compute a headline metric on "
            "a population selected by a different model, which looks like an ordinary "
            "number and is not one. Split the two roles into two objects."
        )
    if not isinstance(explainer, Explainer):
        raise NotAnExplainerError(
            f"{type(explainer).__name__} does not satisfy Explainer: it needs a 'name' "
            "and an 'explain(request) -> str'."
        )
    name = explainer.name
    if name in EXPLAINERS:
        raise ValueError(
            f"explainer {name!r} is already registered by "
            f"{type(EXPLAINERS[name]).__name__}. The name is the join key between this "
            "register and every reason string that cites it — two cannot share one."
        )
    EXPLAINERS[name] = explainer
    return explainer


def get(name: str) -> Explainer:
    try:
        return EXPLAINERS[name]
    except KeyError:
        raise KeyError(
            f"no registered explainer named {name!r}; known: {sorted(EXPLAINERS)}"
        ) from None


def registered() -> tuple[str, ...]:
    """Registered explainer names, in registration order."""
    return tuple(EXPLAINERS)


def reset_for_testing() -> None:
    """Empty the register. Named so its appearance in ``src/`` would be obvious in review,
    exactly as ``features.registry.reset_for_testing`` is."""
    EXPLAINERS.clear()
