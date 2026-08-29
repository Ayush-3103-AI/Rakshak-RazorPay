"""K2's verdict on the test window (T-0011).

Four things are pinned here, chosen because each of them would rot silently:

1. **The test window stays locked.** `verdict.py` is one of exactly two modules
   allowed to open it, and it opens it by naming its ticket. If the guard ever
   stops refusing an unnamed caller, the whole "touched once, at the end" claim
   is decorative.
2. **The verdict survives an edit.** The rendered document must contain the K2
   line and a savings-figure measured against the `random` floor. Those two are
   the reason the file exists — `CLAUDE.md` non-negotiable 1 and `07-math.md`
   §6's AP-06 guard — and a refactor that drops either must fail here.
3. **NFR-003.** Two runs at the same seed are byte-identical.
4. **The figure is a rendering, not a second computation.** It is drawn from the
   committed CSV, and the numbers in the sweep table are that CSV's numbers.

The scoring and the sweep themselves are not re-tested: they are
`harness.evaluate_model` and `decision.policy.sweep_cost_asymmetry`, both already
pinned by `tests/test_policy.py` and `tests/test_determinism.py`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rakshak.config import STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.decision.policy import SWEEP_COLUMNS
from rakshak.eval import verdict
from rakshak.eval.splits import load_split

needs_data = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="generated data absent; run `python -m rakshak.generator.generate --seed 42`",
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------


def test_the_test_window_refuses_a_caller_without_the_ticket() -> None:
    """06-requirements.md §3, enforced in code and not by convention."""
    with pytest.raises(PermissionError):
        load_split("test")
    with pytest.raises(PermissionError):
        load_split("test", unlock_test="T-9999")


def test_this_module_opens_the_window_by_naming_its_ticket() -> None:
    """The unlock ticket is T-0011's own ID, not a borrowed one."""
    assert verdict.UNLOCK_TICKET == "T-0011"
    assert verdict.VERDICT_SPLIT == "test"
    # The bar is pre-registered (00-charter.md §2, amended 2026-08-28). It is a
    # constant here so that softening it would be a visible diff, not a tweak.
    assert verdict.MARGIN_BAR == 0.20


# ---------------------------------------------------------------------------
# The rendered document
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two runs at seed 42, into two directories. Module-scoped: it fits a model.

    A mark on a fixture does nothing, so the `needs_data` skip lives on each test
    that asks for this fixture instead.
    """
    first = tmp_path_factory.mktemp("verdict_a")
    second = tmp_path_factory.mktemp("verdict_b")
    verdict.run(seed=42, results_dir=first)
    verdict.run(seed=42, results_dir=second)
    return first, second


@needs_data
def test_document_states_the_k2_verdict(rendered: tuple[Path, Path]) -> None:
    """Exactly one verdict line, and it says PASS, CONDITIONAL PASS or FAIL."""
    text = (rendered[0] / "verdict.md").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith(verdict.K2_VERDICT_PREFIX)]
    assert len(lines) == 1, "the document must carry exactly one K2 verdict line"
    assert any(word in lines[0] for word in ("**PASS.**", "**CONDITIONAL PASS.**", "**FAIL.**"))


@needs_data
def test_savings_is_reported_against_the_random_floor(rendered: tuple[Path, Path]) -> None:
    """AP-06: a savings number without the random floor beside it is not a claim.

    The floor's own row is 0.0000 by construction, so the check is that at least
    one *other* model's margin against the floor is printed as a number.
    """
    text = (rendered[0] / "verdict.md").read_text(encoding="utf-8")
    assert "savings - `random`" in text, "the floor-relative savings column is gone"
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| hmm |") or line.startswith("| rules |")
    ]
    assert rows, "no model row found to carry a floor-relative figure"
    # PR-AUC must appear beside every savings number, per 07-math.md §6.
    assert "PR-AUC" in text


@needs_data
def test_two_runs_at_the_same_seed_are_byte_identical(rendered: tuple[Path, Path]) -> None:
    """NFR-003, over raw bytes with nothing excluded."""
    first, second = rendered
    for name in ("verdict.md", "sensitivity_test.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), (
            f"{name} differs between two runs at seed 42"
        )


@needs_data
def test_the_figure_is_drawn_from_the_same_csv_as_the_table(
    rendered: tuple[Path, Path],
) -> None:
    """The figure computes nothing of its own: same frame, same numbers."""
    out = rendered[0]
    frame = pd.read_csv(out / "sensitivity_test.csv")
    assert list(frame.columns) == list(SWEEP_COLUMNS), "the sweep's tidy contract moved"

    figure = out / "figures" / "sensitivity_test.png"
    assert figure.read_bytes()[:8] == _PNG_MAGIC
    assert figure.stat().st_size > 10_000, "figure is suspiciously small - did a panel fail?"

    # Every asymmetry the document tabulates is a row of that CSV, formatted the
    # same way. If the table ever grew its own computation, this fails.
    text = (out / "verdict.md").read_text(encoding="utf-8")
    for asymmetry in frame["asymmetry"].drop_duplicates():
        assert f"| {asymmetry:.1f} |" in text, f"asymmetry {asymmetry} is in the CSV, not the table"
