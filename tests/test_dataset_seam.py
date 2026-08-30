"""T-0022b: the dataset seam is inert on the primary path and total off it.

Two things have to hold, and the second is the one the ticket's original `Done
when` list did not ask for:

1. **Inert by default.** `harness.run(seed=42)` with no overrides still writes a
   `summary.md` byte-identical to the committed one.
2. **Total when used.** An override must reach *every* dataset reader in the run,
   including the two inside `gbdt.fit` and `hmm_score.fit`. Those fit through the
   `Scorer` contract, which has no dataset argument, so before T-0022b they read
   `config.STATE_PATHS_PARQUET` directly — meaning a shock-dataset run would have
   scored the shock data with models trained on `data/synthetic/`. Silent
   cross-dataset contamination in exactly the two rows `blackswan.md` is about.
   `test_override_reaches_every_reader` is the guard against that regressing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rakshak.config import (
    RESULTS_DIR,
    STATE_PATHS_PARQUET,
    SYNTHETIC_DIR,
    TRANSACTIONS_PARQUET,
)
from rakshak.eval.harness import run
from rakshak.eval.splits import active_state_paths_path, active_transactions_path

pytestmark = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator.generate --seed 42` first",
)

DROPPED_MERCHANTS = 50
"""How many merchants the contamination copy is missing. Any non-zero number
works; 50 of 500 is large enough to move the printed TOTAL unmistakably."""


def _write_thinned_copy(dest: Path) -> tuple[Path, Path]:
    """Copy the real parquet pair into `dest`, minus `DROPPED_MERCHANTS` merchants.

    Returns:
        `(transactions_path, state_paths_path)` of the copy.
    """
    dest.mkdir(parents=True, exist_ok=True)
    state_paths = pd.read_parquet(STATE_PATHS_PARQUET)
    transactions = pd.read_parquet(TRANSACTIONS_PARQUET)
    keep = sorted(state_paths["merchant_id"].unique())[:-DROPPED_MERCHANTS]
    txn_path = dest / "transactions.parquet"
    path_path = dest / "state_paths.parquet"
    transactions[transactions["merchant_id"].isin(keep)].to_parquet(txn_path, index=False)
    state_paths[state_paths["merchant_id"].isin(keep)].to_parquet(path_path, index=False)
    return txn_path, path_path


def _total_row(summary: str) -> int:
    """Sum the `| TOTAL |` row of the split table the harness prints."""
    for line in summary.splitlines():
        if line.startswith("| TOTAL |"):
            return sum(int(cell) for cell in line.strip("| ").split("|")[1:])
    raise AssertionError("summary.md has no TOTAL row")


def test_no_override_is_byte_identical_to_the_committed_summary(tmp_path: Path) -> None:
    """The seam must be provably inert on the primary path. Written to a tmp
    results_dir and diffed — the committed file is never overwritten to pass."""
    committed = RESULTS_DIR / "summary.md"
    if not committed.exists():
        pytest.skip("results/summary.md not committed yet")
    written = run(seed=42, results_dir=tmp_path / "out")
    assert written.read_bytes() == committed.read_bytes(), (
        "T-0022b changed the primary path: summary.md no longer matches results/summary.md"
    )


def test_override_reaches_every_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The contamination guard. Every parquet read during an overridden run must
    land on the override, including the ones inside `gbdt.fit` / `hmm_score.fit`."""
    txn_path, path_path = _write_thinned_copy(tmp_path / "alt")
    seen: list[Path] = []
    real_read = pd.read_parquet

    def spy(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(Path(path).resolve())
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    written = run(
        seed=42,
        results_dir=tmp_path / "out",
        transactions_path=txn_path,
        state_paths_path=path_path,
    )

    primary = SYNTHETIC_DIR.resolve()
    leaked = [p for p in seen if primary in p.parents]
    assert not leaked, f"overridden run still read the primary dataset: {sorted(set(leaked))}"
    # A run that read nothing at all would pass the check above vacuously.
    assert len(seen) >= 4, f"expected at least 4 parquet reads in a full run, saw {len(seen)}"
    assert _total_row(written.read_text(encoding="utf-8")) == 500 - DROPPED_MERCHANTS


def test_active_paths_are_restored_after_a_run(tmp_path: Path) -> None:
    """The context manager is module-level state; a leak would poison every later
    call in the same process, tests included."""
    txn_path, path_path = _write_thinned_copy(tmp_path / "alt")
    run(seed=42, results_dir=tmp_path / "out", transactions_path=txn_path,
        state_paths_path=path_path)
    assert active_transactions_path() == TRANSACTIONS_PARQUET
    assert active_state_paths_path() == STATE_PATHS_PARQUET
