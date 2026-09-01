"""T-0118 — the explainer register, and the structural wall against scoring.

The acceptance criterion is two-sided: an explainer must be able to register *without*
satisfying ``Scorer``, and a structural test must assert it is not reachable from any
``Scorer``. Both halves are here, and the second one is an import scan rather than a
convention, because "we agreed not to score the HSMM" is not a guarantee.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rakshak.explain.registry import (
    EXPLAINERS,
    Explainer,
    ExplanationRequest,
    NotAnExplainerError,
    Scorer,
    get,
    register,
    registered,
    reset_for_testing,
)
from rakshak.schemas import Action

SRC = Path(__file__).resolve().parents[2] / "src" / "rakshak"


@pytest.fixture(autouse=True)
def _clean_register() -> object:
    reset_for_testing()
    yield
    reset_for_testing()


class ViterbiReasons:
    """A minimal explainer. Note what it does NOT have: ``predict``."""

    @property
    def name(self) -> str:
        return "viterbi_reasons"

    def explain(self, request: ExplanationRequest) -> str:
        return f"{request.merchant_id} scored {request.score:.2f} on day {request.day}"


class ScoringRung:
    """A scorer. It also happens to be able to explain — which is exactly the case the
    register has to refuse, because that is how the HSMM would leak into the scoring
    path."""

    @property
    def name(self) -> str:
        return "hsmm"

    def predict(self, x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        return np.zeros(len(x))

    def explain(self, request: ExplanationRequest) -> str:
        return "state 3 since day 143"


def a_request() -> ExplanationRequest:
    return ExplanationRequest(
        merchant_id="M-0001",
        day=250,
        x=np.zeros(3),
        columns=("a", "b", "c"),
        score=0.91,
        action=Action.REVIEW,
    )


# ─────────────────── an explainer registers without being a Scorer ───────────────────


def test_an_explainer_registers_without_satisfying_the_scorer_contract() -> None:
    explainer = register(ViterbiReasons())
    assert registered() == ("viterbi_reasons",)
    assert get("viterbi_reasons") is explainer
    assert isinstance(explainer, Explainer)
    assert not isinstance(explainer, Scorer)
    assert "M-0001" in explainer.explain(a_request())


def test_every_registered_explainer_is_unreachable_as_a_scorer() -> None:
    """The structural assertion, over the register rather than over one instance."""
    register(ViterbiReasons())
    for name in registered():
        assert not isinstance(get(name), Scorer), f"{name} can score"
        assert not hasattr(get(name), "predict")


def test_something_that_can_score_is_refused() -> None:
    with pytest.raises(NotAnExplainerError, match="satisfies Scorer"):
        register(ScoringRung())
    assert registered() == ()


def test_something_that_cannot_explain_is_refused() -> None:
    class Nothing:
        pass

    with pytest.raises(NotAnExplainerError, match="does not satisfy Explainer"):
        register(Nothing())  # type: ignore[arg-type]


def test_a_duplicate_name_is_refused() -> None:
    register(ViterbiReasons())
    with pytest.raises(ValueError, match="already registered"):
        register(ViterbiReasons())


def test_an_unknown_name_lists_what_is_known() -> None:
    register(ViterbiReasons())
    with pytest.raises(KeyError, match="viterbi_reasons"):
        get("nope")


def test_a_fresh_import_registers_nothing() -> None:
    """Pre-registration §4: the surface lands with no rung attached.

    Run in a subprocess because this module's autouse fixture has already emptied
    ``EXPLAINERS`` in-process, so asserting on it here would prove nothing about what
    importing the package actually does.
    """
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rakshak.explain.registry as r; print(sorted(r.EXPLAINERS))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]", out.stdout
    assert EXPLAINERS == {}


# ──────────────────────── the wall, checked by import scan ────────────────────────


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_scoring_module_can_reach_the_explain_package() -> None:
    """Reachability, not intent: if ``models/`` cannot import ``explain``, an explainer
    cannot become a scoring rung by accident however tempting its posterior looks."""
    offenders = [
        str(p.relative_to(SRC))
        for p in (SRC / "models").rglob("*.py")
        if any(m.startswith("rakshak.explain") for m in _imports(p))
    ]
    assert offenders == [], f"scoring modules importing rakshak.explain: {offenders}"


def test_the_explain_package_does_not_import_the_models_package() -> None:
    """And not the other way either — the ``Scorer`` protocol here is structural, so the
    wall does not need a dependency to stand."""
    offenders = [
        str(p.relative_to(SRC))
        for p in (SRC / "explain").rglob("*.py")
        if any(m.startswith("rakshak.models") for m in _imports(p))
    ]
    assert offenders == []
