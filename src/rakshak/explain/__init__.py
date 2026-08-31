"""Rakshak explain layer — the Viterbi path rendered as a sentence (FR-014).

`models/hmm.py` produces a MAP state path; this package turns it into the string
a risk-ops analyst reads to a merchant who has called to shout about a held
settlement. `CLAUDE.md` names it the centrepiece, and it is the only part of the
repo that answers the audience's third question.

See `rakshak.explain.reasons` for the mathematics — the decomposition is exact
because the emission covariance is diagonal — and for why the decode is truncated
at the flag window.
"""

from __future__ import annotations

from rakshak.explain.reasons import (
    REASONS_SPLIT,
    TOP_N_FEATURES,
    FeatureContribution,
    Reason,
    build_reasons,
    emission_contributions,
    explain_merchant,
    render_json,
    run,
)

__all__ = [
    "REASONS_SPLIT",
    "TOP_N_FEATURES",
    "FeatureContribution",
    "Reason",
    "build_reasons",
    "emission_contributions",
    "explain_merchant",
    "render_json",
    "run",
]
