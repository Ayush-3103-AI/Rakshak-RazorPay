"""Viterbi path -> merchant-readable reason string (FR-014, T-0013).

`CLAUDE.md` names this the centrepiece and the only answer to the audience's
third question — *"Can I explain the decision to the merchant when they call and
shout?"* Every other module in this repo produces a number; this one produces a
sentence a risk-ops analyst can read down a phone line.

**Where the explanation comes from, exactly.** The emission model is a Gaussian
with **diagonal** covariance (`models/hmm.py`, 07-math.md §1), so its
log-density is already a plain sum over features::

    log N(x | mu_k, diag(var_k)) = sum_d -0.5 * [ log(2 pi var_kd) + (x_d - mu_kd)^2 / var_kd ]

That makes the per-feature decomposition **exact rather than attributed**: the
contribution of feature *d* to preferring state `new` over state `prev` is the
difference of the two per-feature terms, and those D differences sum to the total
log-likelihood ratio with no residual. There is no sampling, no surrogate model
and no approximation anywhere in this file, which is the whole reason the HMM was
worth hand-writing. `tests/test_reasons.py::test_contributions_sum_to_the_log_ratio`
pins the identity.

**The explanation uses no information the flag did not have.** `hmm_score` raises
its flag from the *forward-only* filtered posterior. A Viterbi decode over the
whole sequence would condition on windows after the flag, so this module decodes
the **truncated** sequence ``X[:flag_index + 1]`` and nothing longer. The reason
string therefore rests on exactly the evidence available on the day the flag
fired. `tests/test_reasons.py::test_explanation_ignores_the_future` is the proof.

**The MAP path and the filtered flag can disagree, and the disagreement ships.**
The flag fires when the summed filtered probability of the three bad states
reaches `FLAG_THRESHOLD`; the Viterbi path commits to a single most-probable
joint sequence. Belief spread thinly across RAMP, FRAUD and DORMANT can cross 0.5
in aggregate while the MAP path still sits in HEALTHY. When that happens the
reason carries ``viterbi_agrees: false`` and names the highest-posterior bad state
instead of a decoded transition. This is reported per merchant and counted in the
provenance block rather than being smoothed over — a reason string that quietly
invented a transition the model never decoded would be worse than no reason
string at all.

**Units.** Emissions reaching this module are within-merchant z-scores (FR-007),
so a contribution is in **nats** and a reported level is in **standard deviations
of that merchant's own baseline**. The strings say so. They do not say "6x above
your norm", because the model never saw a ratio.

Run it::

    python -m rakshak.explain.reasons --seed 42

which writes `results/reasons.json` — the machine-readable half T-0014's viewer
renders directly, so the dashboard cannot hand-transcribe a number that already
lives here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    GENERATOR_START_DATE,
    RESULTS_DIR,
    SEED,
    WINDOW_DAYS,
    Z_CLIP,
)
from rakshak.eval.splits import Split, load_split
from rakshak.models.gbdt import build_window_matrix, decision_mask
from rakshak.models.hmm import HMM
from rakshak.models.hmm_score import (
    BAD_COLUMNS,
    FLAG_THRESHOLD,
    STATE_ORDER,
    _fitted,
    _panel,
    filtered_bad_probability,
    first_flag_index,
)

UNLOCK_TICKET: str = "T-0013"
"""`eval.splits.load_split` requires this to open the test window. T-0011 and
T-0013 are the only two tickets 06-requirements.md §3 authorises."""

REASONS_SPLIT: str = "test"
"""Reasons are rendered for the window the verdict reports on, so the strings and
`results/verdict.md` describe the same merchants under the same fitted model."""

TOP_N_FEATURES: int = 3
"""FR-014: "the top 3 emissions by contribution to the transition"."""

HEALTHY_INDEX: int = STATE_ORDER.index("HEALTHY")
"""Fallback contrast state when a bad run starts at the very first window and
there is no decoded predecessor to contrast against."""

_MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
"""Explicit English month names. `datetime.strftime('%B')` is locale-dependent,
which would make `results/reasons.json` differ between machines and quietly break
NFR-003's byte-identity guarantee on any non-English host."""

PATTERN_LABELS: dict[str, str] = {
    "RAMP": "rapid-volume-escalation",
    "FRAUD": "sustained-abnormal-activity",
    "DORMANT": "abrupt-stop-after-escalation",
}
"""Latent state -> the phrase the merchant hears. The generator's state names are
internal vocabulary; "FRAUD" is not a word to put in front of a merchant who may
well be innocent, and at this repo's measured precision many of them are."""

REMEDIATION: dict[str, str] = {
    "RAMP": (
        "recent invoices or purchase orders covering the increase, and the campaign "
        "or sales channel that drove it"
    ),
    "FRAUD": (
        "settlement-level invoices for the flagged period, and contact details for "
        "the payers behind the largest transactions"
    ),
    "DORMANT": (
        "confirmation of whether the account is still trading, and settlement "
        "instructions for the outstanding balance"
    ),
}
"""What the merchant can send to resolve the flag. FR-014's example string ends
with this clause, and it is the half that turns a detection into a conversation."""

FEATURE_PHRASES: dict[str, str] = {
    "log_amount_mean": "average ticket size",
    "log_amount_var": "spread of ticket sizes",
    "log_velocity": "transaction volume",
    "refund_ratio": "refund rate",
    "chargeback_ratio": "chargeback rate",
    "chargeback_lag_days": "time from a payer's first payment to their chargeback",
    "hour_entropy": "spread of payments across the hours of the day",
    "method_entropy": "mix of payment methods",
    "new_payer_ratio": "share of first-time payers",
    "payer_entropy": "spread of payments across the payer base",
    "repeat_payer_ratio": "share of repeat payers",
    "payer_jaccard_prev": "overlap between this week's payers and last week's",
    "payer_herfindahl": "concentration of payment value in a few payers",
    "sparse": "days with no transactions at all",
    "vulcan_mean": "average per-transaction risk score",
    "vulcan_p95": "worst-case per-transaction risk score",
}
"""Emission name -> merchant-facing noun phrase. A feature with no entry renders
under its own name rather than being dropped, so an added feature degrades the
sentence instead of vanishing from the explanation."""

__all__ = [
    "REASONS_SPLIT",
    "TOP_N_FEATURES",
    "UNLOCK_TICKET",
    "FeatureContribution",
    "Reason",
    "build_reasons",
    "emission_contributions",
    "explain_merchant",
    "log_emission_per_feature",
    "main",
    "render_json",
    "run",
]


# ---------------------------------------------------------------------------
# The decomposition — exact, because the covariance is diagonal
# ---------------------------------------------------------------------------


def log_emission_per_feature(model: HMM, x: np.ndarray) -> np.ndarray:
    """Per-feature log-density of one observation under each state.

    Summing the returned array over its feature axis reproduces
    `HMM.log_emission` for the same observation, exactly — that identity is what
    makes the contributions below a decomposition rather than an attribution.

    Args:
        model: A fitted HMM.
        x: One observation, shape (D,). Dimensionless z-scores (FR-007).

    Returns:
        Array of shape (K, D) in nats: entry (k, d) is the contribution of
        feature *d* to ``log N(x | mu_k, diag(var_k))``.
    """
    deviation_squared = (np.asarray(x, dtype=float)[None, :] - model.mu) ** 2
    return -0.5 * (np.log(2.0 * np.pi * model.var) + deviation_squared / model.var)


def emission_contributions(
    model: HMM, x: np.ndarray, new_state: int, previous_state: int
) -> np.ndarray:
    """How much each feature favours `new_state` over `previous_state` at `x`.

    Args:
        model: A fitted HMM.
        x: The observation at the transition window, shape (D,).
        new_state: Index of the state moved into.
        previous_state: Index of the state moved out of.

    Returns:
        Array of shape (D,) in nats. Positive entries are evidence *for* the
        transition; the entries sum to the total emission log-likelihood ratio
        between the two states at this window.
    """
    per_feature = log_emission_per_feature(model, x)
    return per_feature[new_state] - per_feature[previous_state]


# ---------------------------------------------------------------------------
# The reason, as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContribution:
    """One emission's share of a decoded transition.

    Attributes:
        feature: Emission name, as it appears in `WindowMatrix.feature_names`.
        phrase: The merchant-facing rendering of `feature`.
        z: The standardised level at the transition window. Units: standard
            deviations of this merchant's own baseline (FR-007). Positive is
            above baseline.
        contribution: Evidence for the transition carried by this feature alone.
            Units: nats.
    """

    feature: str
    phrase: str
    z: float
    contribution: float


@dataclass(frozen=True)
class Reason:
    """The full explanation for one flagged merchant.

    Attributes:
        merchant_id: The merchant this explains.
        flag_day: Day the flag fired — the last day of the window whose evidence
            raised it, matching `hmm_score.first_flag_day`. Units: days since
            `GENERATOR_START_DATE`.
        flag_date: `flag_day` as a calendar date.
        transition_day: Last day of the window the decoded transition happened
            in. Equals `flag_day` when `viterbi_agrees` is False. Units: days.
        transition_date: `transition_day` as a calendar date.
        state: The latent state moved into when the account left HEALTHY, from
            `STATE_ORDER`.
        pattern: `state` rendered as the merchant-facing label.
        state_at_flag: The decoded state at the flag window. Differs from `state`
            when the account escalated between leaving HEALTHY and being flagged
            (RAMP -> FRAUD is the common case); the rendered string says so.
        previous_state: The state moved out of — HEALTHY by construction, since
            it is the only state not in `BAD_COLUMNS`.
        viterbi_agrees: True when the truncated Viterbi path is in a bad state at
            the flag window, i.e. the MAP path corroborates the filtered flag.
            False means the belief crossed the threshold in aggregate while the
            MAP path stayed healthy — see the module docstring.
        bad_probability: Filtered bad-state probability at the flag window.
            Dimensionless, [0, 1].
        top_features: Up to `TOP_N_FEATURES` positive contributions, largest
            first. Empty when the transition was driven by the transition prior
            rather than by any emission.
        text: The rendered merchant-facing string (FR-014).
    """

    merchant_id: str
    flag_day: int
    flag_date: str
    transition_day: int
    transition_date: str
    state: str
    pattern: str
    state_at_flag: str
    previous_state: str
    viterbi_agrees: bool
    bad_probability: float
    top_features: tuple[FeatureContribution, ...]
    text: str


def _calendar_date(day: int) -> str:
    """Day index -> "27 August 2026". Locale-independent by construction."""
    stamp = pd.Timestamp(GENERATOR_START_DATE) + pd.Timedelta(days=int(day))
    return f"{stamp.day} {_MONTHS[stamp.month - 1]} {stamp.year}"


def _magnitude(z: float) -> str:
    """Plain-language size of a standardised deviation."""
    size = abs(z)
    if size >= 3.0:
        return "far"
    if size >= 1.5:
        return "well"
    return "somewhat"


def _render(reason_fields: dict[str, object], features: tuple[FeatureContribution, ...]) -> str:
    """Compose the merchant-facing sentence (FR-014's acceptance shape)."""
    opening = (
        f"On {reason_fields['transition_date']} this account moved into a pattern we flag "
        f"as {reason_fields['pattern']}."
    )
    if not reason_fields["viterbi_agrees"]:
        opening = (
            f"On {reason_fields['transition_date']} our confidence that this account had "
            f"left normal operation passed the review threshold, with "
            f"{reason_fields['pattern']} the most likely explanation."
        )

    if features:
        clauses = [
            (
                # At the winsorisation limit the number is a floor, not a level.
                # Printing "+10.00 SD" as though it were measured would overstate
                # what `features/standardise.py` actually kept (`config.Z_CLIP`).
                f"{item.phrase} ran beyond the range we can measure against this "
                f"account's own baseline (at or {'above' if item.z >= 0 else 'below'} "
                f"{item.z:+.2f} SD, where the scale is capped)"
                if abs(item.z) >= Z_CLIP
                else f"{item.phrase} ran {_magnitude(item.z)} "
                f"{'above' if item.z >= 0 else 'below'} this account's own baseline "
                f"({item.z:+.2f} SD)"
            )
            for item in features
        ]
        body = (
            "The measurements that moved it there, each compared against this account's "
            "own trading history rather than against other merchants: "
            + "; ".join(clauses)
            + "."
        )
    else:
        body = (
            "No single measurement drove this on its own — it follows from the sequence "
            "of the preceding weeks taken together."
        )

    # The account may have escalated between leaving HEALTHY and being flagged.
    # Saying so is the difference between an explanation and a half-truth.
    # Neutral verb deliberately: the decoded step can be RAMP -> FRAUD or
    # FRAUD -> RAMP, and "progressed" would assert a severity direction the model
    # never ordered its states by.
    escalation = ""
    if reason_fields["state_at_flag"] != reason_fields["state"]:
        escalation = (
            f"By {reason_fields['flag_date']} the pattern had changed to "
            f"{PATTERN_LABELS[str(reason_fields['state_at_flag'])]}."
        )

    # Keyed on the current state, not the entry state: the merchant needs to
    # resolve where the account is now.
    remedy = REMEDIATION.get(str(reason_fields["state_at_flag"]), "")
    closing = f"What would resolve this: {remedy}." if remedy else ""
    return " ".join(part for part in (opening, body, escalation, closing) if part)


# ---------------------------------------------------------------------------
# Explaining one merchant
# ---------------------------------------------------------------------------


def explain_merchant(
    model: HMM,
    merchant_id: str,
    sequence: np.ndarray,
    eligible: np.ndarray,
    window_start_day: np.ndarray,
    feature_names: tuple[str, ...],
) -> Reason | None:
    """Explain one merchant's flag, or return None if it was never flagged.

    The Viterbi decode runs on ``sequence[:flag_index + 1]`` and nothing longer,
    so the explanation rests on exactly the evidence the flag had.

    Args:
        model: The fitted HMM whose flag is being explained.
        merchant_id: The merchant.
        sequence: This merchant's emissions, shape (T, D), z-scores.
        eligible: True where the window lies in the split's decision window,
            shape (T,).
        window_start_day: Absolute start day of each window, shape (T,). Days.
        feature_names: Emission names in column order, length D.

    Returns:
        A `Reason`, or None if no eligible window reached `FLAG_THRESHOLD`.
    """
    probability = filtered_bad_probability(model, sequence)
    flag_index = first_flag_index(probability, eligible)
    if flag_index is None:
        return None

    # Truncated: the decode may not see a window the flag did not see.
    path = model.viterbi(sequence[: flag_index + 1])
    decoded = int(path[flag_index])
    viterbi_agrees = decoded in BAD_COLUMNS

    if viterbi_agrees:
        # Walk back to where the account LEFT HEALTHY, not merely to the last state
        # change. A merchant that decoded RAMP -> FRAUD -> RAMP has one event worth
        # explaining — the departure from normal operation — and contrasting the
        # emissions against a preceding *bad* state would answer a question the
        # merchant did not ask ("why RAMP rather than FRAUD") while reading as an
        # escalation when the decoded step was a de-escalation. The only non-bad
        # state is HEALTHY, so `previous_state` below is HEALTHY by construction.
        transition_index = flag_index
        while transition_index > 0 and int(path[transition_index - 1]) in BAD_COLUMNS:
            transition_index -= 1
        new_state = int(path[transition_index])
        previous_state = (
            int(path[transition_index - 1]) if transition_index > 0 else HEALTHY_INDEX
        )
    else:
        # The MAP path never committed. Name the likeliest bad state and say so.
        log_alpha, _ = model.forward(sequence[: flag_index + 1])
        posterior = np.exp(log_alpha[flag_index] - log_alpha[flag_index].max())
        new_state = int(max(BAD_COLUMNS, key=lambda k: posterior[k]))
        previous_state = decoded
        transition_index = flag_index

    contributions = emission_contributions(
        model, sequence[transition_index], new_state, previous_state
    )
    # Stable sort on the negated array, NOT `argsort(...)[::-1]`: the default
    # quicksort orders ties arbitrarily, so two features with equal contribution
    # could swap places between numpy versions and change a committed artifact.
    # Stable + negated breaks ties by feature index, which is fixed.
    order = np.argsort(-contributions, kind="stable")[:TOP_N_FEATURES]
    top_features = tuple(
        FeatureContribution(
            feature=feature_names[d],
            phrase=FEATURE_PHRASES.get(feature_names[d], feature_names[d]),
            z=round(float(sequence[transition_index, d]), 6),
            contribution=round(float(contributions[d]), 6),
        )
        for d in order
        if contributions[d] > 0.0
    )

    # Both days are window-END, the convention `hmm_score.first_flag_day` fixed at
    # T-0011: a window's evidence is not complete until its final day.
    flag_day = int(window_start_day[flag_index]) + WINDOW_DAYS - 1
    transition_day = int(window_start_day[transition_index]) + WINDOW_DAYS - 1

    state_at_flag = STATE_ORDER[decoded] if viterbi_agrees else STATE_ORDER[new_state]
    fields: dict[str, object] = {
        "transition_date": _calendar_date(transition_day),
        "flag_date": _calendar_date(flag_day),
        "pattern": PATTERN_LABELS[STATE_ORDER[new_state]],
        "state": STATE_ORDER[new_state],
        "state_at_flag": state_at_flag,
        "viterbi_agrees": viterbi_agrees,
    }
    return Reason(
        merchant_id=merchant_id,
        flag_day=flag_day,
        flag_date=_calendar_date(flag_day),
        transition_day=transition_day,
        transition_date=_calendar_date(transition_day),
        state=STATE_ORDER[new_state],
        pattern=PATTERN_LABELS[STATE_ORDER[new_state]],
        state_at_flag=state_at_flag,
        previous_state=STATE_ORDER[previous_state],
        viterbi_agrees=viterbi_agrees,
        bad_probability=round(float(probability[flag_index]), 6),
        top_features=top_features,
        text=_render(fields, top_features),
    )


# ---------------------------------------------------------------------------
# Explaining a split
# ---------------------------------------------------------------------------


def build_reasons(seed: int = SEED, split: Split | None = None) -> list[Reason]:
    """Explain every flagged merchant on the test split.

    The fit seed is derived exactly as `harness.evaluate_model` derives it for the
    `hmm` row — `_model_rng(seed, "hmm")` then one `integers` draw — so these
    strings explain the *same fitted model* that produced the verdict table, not a
    second fit that happens to share a global seed.

    Args:
        seed: Global determinism seed (NFR-003).
        split: Pre-loaded split, for tests. None loads `test` with the unlock.

    Returns:
        Reasons for the flagged merchants, ordered by merchant id.
    """
    # Imported here: `eval.harness` imports the model modules, and importing it at
    # module scope would close an import cycle through `models.hmm_score`.
    from rakshak.eval.harness import _model_rng

    if split is None:
        split = load_split(REASONS_SPLIT, unlock_test=UNLOCK_TICKET)

    fit_seed = int(_model_rng(seed, "hmm").integers(0, 2**31 - 1))
    model, segment_map, feature_names = _fitted(fit_seed)

    matrix = build_window_matrix(split, segment_map=segment_map)
    sequences = _panel(matrix)
    n_merchants, n_windows, _ = sequences.shape
    eligible = decision_mask(matrix, split).reshape(n_merchants, n_windows)
    days = matrix.window_start_day.reshape(n_merchants, n_windows)

    reasons = [
        explain_merchant(
            model,
            str(matrix.merchant_ids[i]),
            sequences[i],
            eligible[i],
            days[i],
            feature_names,
        )
        for i in range(n_merchants)
    ]
    return sorted(
        (reason for reason in reasons if reason is not None),
        key=lambda reason: reason.merchant_id,
    )


def render_json(reasons: list[Reason], seed: int, split: Split) -> str:
    """Serialise reasons to the frozen `results/reasons.json` shape.

    The shape is frozen here, not in T-0014: the viewer renders this file and must
    not redesign it. Every float is rounded at construction so the file is
    byte-identical across platforms for a fixed seed (NFR-003).

    Args:
        reasons: Reasons to write, already ordered.
        seed: The seed that produced them, recorded as provenance.
        split: The split they were built on, recorded as provenance.

    Returns:
        The JSON document as a string, newline-terminated.
    """
    n_bad = int(split.labels.sum())
    flagged = {reason.merchant_id for reason in reasons}
    true_positives = int(
        sum(1 for m in flagged if bool(split.labels.get(m, 0)))
    )
    disagreements = sum(1 for reason in reasons if not reason.viterbi_agrees)
    document = {
        "provenance": {
            "produced_by": f"python -m rakshak.explain.reasons --seed {seed}",
            "seed": seed,
            "split": split.name,
            "split_days": [split.start_day, split.end_day - 1],
            "model": "hmm",
            "requirement": "FR-014",
            "generator_start_date": GENERATOR_START_DATE,
            "flag_threshold": FLAG_THRESHOLD,
            "contribution_units": "nats",
            "level_units": "standard deviations of the merchant's own baseline",
            "day_convention": (
                "window-END: the last day of the window whose evidence raised the flag"
            ),
            "transition_day_note": (
                "transition_day may precede the reported split window. It is the window "
                "in which the MAP path left HEALTHY, decoded over the merchant's own "
                "prior history, which the model legitimately observes when filtering. "
                "It is a DECODE, not a detection: the detection is flag_day. The gap "
                "between them is not evidence of early warning and must not be quoted "
                "as detection lag - see results/lag_probe.md for the measured lag."
            ),
        },
        "counts": {
            "merchants_in_split": split.n_merchants,
            "merchants_truly_bad": n_bad,
            "merchants_flagged": len(reasons),
            "flagged_and_truly_bad": true_positives,
            "flagged_and_healthy": len(reasons) - true_positives,
            "viterbi_disagreed_with_flag": disagreements,
        },
        "reasons": [asdict(reason) for reason in reasons],
    }
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def run(seed: int = SEED, results_dir: Path = RESULTS_DIR) -> Path:
    """Build the reasons and write `results/reasons.json`. Returns its path."""
    split = load_split(REASONS_SPLIT, unlock_test=UNLOCK_TICKET)
    reasons = build_reasons(seed, split=split)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "reasons.json"
    path.write_text(render_json(reasons, seed, split), encoding="utf-8", newline="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """Write `results/reasons.json`. Returns a process exit code."""
    parser = base_parser("Render FR-014 reason strings from the HMM's Viterbi path.")
    args = parser.parse_args(argv)
    seed_everything(args.seed)

    path = run(args.seed)
    document = json.loads(path.read_text(encoding="utf-8"))
    counts = document["counts"]
    print(f"rakshak: wrote {path} (seed={args.seed})")
    print(
        f"rakshak: {counts['merchants_flagged']} merchants flagged on the "
        f"{document['provenance']['split']} split "
        f"({counts['flagged_and_truly_bad']} truly bad, "
        f"{counts['flagged_and_healthy']} healthy)"
    )
    print(
        f"rakshak: the MAP path disagreed with the filtered flag on "
        f"{counts['viterbi_disagreed_with_flag']} of them"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
