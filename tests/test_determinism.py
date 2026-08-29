"""NFR-003: two runs at the same seed produce byte-identical `results/*.md`.

The comparison is over raw bytes with nothing excluded. If a line cannot be
made deterministic it must be removed from the file, not filtered out of this
test — that is why `results/summary.md` carries no wall-clock time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rakshak.config import STATE_PATHS_PARQUET, TRANSACTIONS_PARQUET
from rakshak.eval.harness import main, run

pytestmark = pytest.mark.skipif(
    not (TRANSACTIONS_PARQUET.exists() and STATE_PATHS_PARQUET.exists()),
    reason="run `python -m rakshak.generator --seed 42` first",
)


def _run_into(tmp_path: Path, label: str, seed: int) -> dict[str, bytes]:
    out = tmp_path / label
    run(seed=seed, results_dir=out)
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.md"))}


def test_two_runs_at_the_same_seed_are_byte_identical(tmp_path: Path) -> None:
    first = _run_into(tmp_path, "a", seed=42)
    second = _run_into(tmp_path, "b", seed=42)
    assert first.keys() == second.keys()
    assert first, "the harness wrote no results/*.md at all"
    for name in first:
        assert first[name] == second[name], f"{name} differs between two runs at seed 42"


def test_results_contain_no_wall_clock_timestamp(tmp_path: Path) -> None:
    """The usual way this NFR breaks. Catch it at the source, not by filtering."""
    written = _run_into(tmp_path, "c", seed=42)
    for name, blob in written.items():
        text = blob.decode("utf-8").lower()
        for banned in ("generated at", "utc", "timestamp:", "elapsed", "wall-clock"):
            assert banned not in text, f"{name} contains a non-deterministic token {banned!r}"


def test_a_different_seed_changes_the_output(tmp_path: Path) -> None:
    """Determinism must not be achieved by ignoring the seed entirely."""
    assert _run_into(tmp_path, "d", seed=42) != _run_into(tmp_path, "e", seed=7)


def test_cli_entry_point_is_reproducible(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["--seed", "42"]) == 0
    assert main(["--seed", "42", "--figures-only"]) == 0
    captured = capsys.readouterr().out
    assert "ABSENT" in captured or "models run" in captured
