"""T-0006b — the HMM scorer, and the proof that `flag_day` cannot see the future.

The load-bearing test in this module is `test_filtered_posterior_ignores_the_future`.
It asserts that the scoring quantity at window *t* is **bitwise** unchanged when every
window after *t* is truncated away, for several *t* across several merchants. Bitwise
prefix equality is the strong form of the claim: `flag_day` is a deterministic
elementwise function of that vector, so if the vector's prefix cannot move, neither can
the flag decision at any window in it.

`test_the_truncation_test_has_teeth` is the negative control. It runs the identical
assertion against `HMM.posterior` — the smoothed forward-backward posterior — and
requires it to FAIL. Without that control the truncation test could be passing because
it is vacuous rather than because `hmm_score` is correct, and a panel member probing
this exact hole would be right to ask.

No test here asserts that the HMM beats anything. 06-requirements.md §3 puts the
comparison at T-0011, on the test window, which this module never opens.

Numbers touched here are measured on SYNTHETIC merchant streams with injected
typologies; the generator is in this repo (CLAUDE.md non-negotiable #3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rakshak.config import (
    N_HIDDEN_STATES,
    SEED,
    STATE_PATHS_PARQUET,
    TRANSACTIONS_PARQUET,
)
from rakshak.eval.harness import MODEL_REGISTRY, _model_rng, _normalise, evaluate_model
from rakshak.eval.splits import Split, load_split
from rakshak.models import hmm_score
from rakshak.models.hmm import HMM

needs_data = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator --seed 42` first",
)

TRUNCATION_POINTS: tuple[int, ...] = (2, 5, 9, 14, 20, 25)
"""Windows to truncate at. Several, not one: an off-by-one in a single recursion
step would survive a test that only ever cuts in the middle of the sequence."""


def _toy_model_and_sequences() -> tuple[HMM, list[np.ndarray]]:
    """A fitted HMM and some sequences, with no dependency on generator output.

    Deliberately runs without the parquet files so the central guarantee of this
    ticket is always exercised, including on a fresh clone.
    """
    rng = np.random.default_rng(11)
    sequences = [rng.normal(size=(30, 3)) for _ in range(6)]
    model = HMM(n_states=N_HIDDEN_STATES, n_features=3)
    model.fit(sequences, rng=np.random.default_rng(12))
    return model, sequences


# ---------------------------------------------------------------------------
# THE CONSTRAINT — flag_day must not see the future
# ---------------------------------------------------------------------------


def test_filtered_posterior_ignores_the_future() -> None:
    """Truncating every window after *t* must not move the belief at or before *t*.

    Bitwise, via `assert_array_equal`, not `assert_allclose`: a smoothed quantity
    would differ in the third decimal, not the last bit, so a tolerance here would
    let the exact bug this ticket exists to prevent through.
    """
    model, sequences = _toy_model_and_sequences()
    for row, X in enumerate(sequences):
        full = hmm_score.filtered_bad_probability(model, X)
        for t in TRUNCATION_POINTS:
            truncated = hmm_score.filtered_bad_probability(model, X[: t + 1])
            np.testing.assert_array_equal(
                truncated,
                full[: t + 1],
                err_msg=(
                    f"merchant row {row}: the filtered belief up to window {t} moved when "
                    f"windows after {t} were removed — the scoring path is reading the "
                    "future (T-0006b)"
                ),
            )


def test_flag_day_is_unchanged_when_later_windows_are_truncated() -> None:
    """The claim stated on the reported quantity itself, not only on its input."""
    model, sequences = _toy_model_and_sequences()
    days = np.arange(30) * 7
    eligible = np.ones(30, dtype=bool)
    for row, X in enumerate(sequences):
        full_probability = hmm_score.filtered_bad_probability(model, X)
        full_flag = hmm_score.first_flag_day(full_probability, eligible, days)
        for t in TRUNCATION_POINTS:
            truncated = hmm_score.first_flag_day(
                hmm_score.filtered_bad_probability(model, X[: t + 1]),
                eligible[: t + 1],
                days[: t + 1],
            )
            # A flag the truncated run cannot yet have seen is the only permitted
            # difference; anything the truncated run does see must be identical.
            expected = (
                full_flag if (full_flag == full_flag and full_flag <= days[t]) else float("nan")
            )
            if expected != expected:
                assert truncated != truncated, (
                    f"merchant row {row}: truncating at window {t} produced flag day "
                    f"{truncated}, but the full sequence does not flag until {full_flag}"
                )
            else:
                assert truncated == expected, (
                    f"merchant row {row}: flag day moved from {expected} to {truncated} "
                    f"when windows after {t} were removed"
                )


def test_the_truncation_test_has_teeth() -> None:
    """The negative control: the SMOOTHED posterior must fail the assertion above.

    If this test ever starts failing it means the truncation assertion has become
    vacuous — the two posteriors have stopped differing — and the guarantee the
    other tests claim to prove would no longer be proven by them.
    """
    model, sequences = _toy_model_and_sequences()
    smoothed_moved = False
    for X in sequences:
        full = model.posterior(X)[:, hmm_score.BAD_COLUMNS].sum(axis=1)
        for t in TRUNCATION_POINTS:
            truncated = model.posterior(X[: t + 1])[:, hmm_score.BAD_COLUMNS].sum(axis=1)
            if not np.array_equal(truncated, full[: t + 1]):
                smoothed_moved = True
    assert smoothed_moved, (
        "the smoothed forward-backward posterior did not move under truncation, so the "
        "truncation test cannot distinguish it from the filtered one and proves nothing"
    )


@needs_data
def test_filtered_posterior_ignores_the_future_on_real_emissions() -> None:
    """The same proof on the real fitted model and the real validate emissions."""
    split = load_split("validate")
    model, segment_map, _ = hmm_score._fitted(SEED)
    matrix = hmm_score.build_window_matrix(split, segment_map=segment_map)
    sequences = hmm_score._panel(matrix)

    for row in range(0, sequences.shape[0], 17):  # a spread of merchants, not just the first
        X = sequences[row]
        full = hmm_score.filtered_bad_probability(model, X)
        for t in TRUNCATION_POINTS:
            np.testing.assert_array_equal(
                hmm_score.filtered_bad_probability(model, X[: t + 1]),
                full[: t + 1],
                err_msg=f"merchant {matrix.merchant_ids[row]} at window {t}",
            )


@needs_data
def test_scoring_path_never_calls_the_smoothed_posterior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HMM.posterior` must be unreachable from the whole end-to-end scorer.

    The truncation tests prove the output is clean for the one function they call;
    this one blows up `HMM.posterior` and runs `score_hmm` against the real
    validate split, so a future edit that reintroduces forward-backward anywhere in
    the scoring path fails loudly rather than shifting the lag by a window.
    """

    def forbidden(self: HMM, X: np.ndarray) -> np.ndarray:
        raise AssertionError("hmm_score called HMM.posterior — flag_day would see the future")

    monkeypatch.setattr(HMM, "posterior", forbidden)
    frame = hmm_score.score_hmm(load_split("validate"), _model_rng(SEED, "hmm"))
    assert frame["score"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Harness contract (T-0006b "Done when")
# ---------------------------------------------------------------------------


def test_hmm_is_registered_and_not_absent() -> None:
    from rakshak.eval.harness import EXPECTED_MODELS

    assert "hmm" in MODEL_REGISTRY
    absent = [name for name, _ in EXPECTED_MODELS if name not in MODEL_REGISTRY]
    assert "hmm" not in absent


def test_bad_columns_match_the_configured_bad_states() -> None:
    """The score sums exactly the states `results/summary.md` calls bad."""
    from rakshak.config import BAD_STATES

    assert {hmm_score.STATE_ORDER[k] for k in hmm_score.BAD_COLUMNS} == set(BAD_STATES)


@needs_data
def test_hmm_produces_a_valid_harness_row() -> None:
    split = load_split("validate")
    frame = _normalise(MODEL_REGISTRY["hmm"](split, _model_rng(SEED, "hmm")), split)
    assert list(frame.index) == list(split.merchant_ids)
    assert frame["score"].notna().all()
    assert frame["score"].between(0.0, 1.0).all()
    assert frame["flag_day"].notna().any(), "the HMM flagged nobody; median lag would be n/a"

    row = evaluate_model("hmm", split, seed=SEED, k=5)
    assert row["model"] == "hmm"
    assert 0.0 <= float(row["pr_auc"]) <= 1.0
    assert 0.0 <= float(row["precision_at_k"]) <= 1.0
    assert 0.0 <= float(row["brier"]) <= 1.0
    assert row["n_reviewed"] == 5


@needs_data
def test_hmm_is_deterministic_at_a_fixed_seed() -> None:
    """NFR-003 on the model path: same seed, byte-identical scores."""
    split = load_split("validate")
    first = _normalise(MODEL_REGISTRY["hmm"](split, _model_rng(SEED, "hmm")), split)
    second = _normalise(MODEL_REGISTRY["hmm"](split, _model_rng(SEED, "hmm")), split)
    pd.testing.assert_frame_equal(first, second)


@needs_data
def test_hmm_fitting_never_opens_the_test_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The leakage guard for this ticket, mirroring the one `gbdt` carries."""
    opened: list[str] = []
    real = hmm_score.load_split

    def spy(name: str, **kwargs: object) -> Split:
        opened.append(name)
        assert "unlock_test" not in kwargs, "a scorer may not unlock the test window"
        return real(name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(hmm_score, "load_split", spy)
    hmm_score.fit(seed=SEED)
    assert set(opened) == {"train"}, f"the fit opened {sorted(set(opened))}"
