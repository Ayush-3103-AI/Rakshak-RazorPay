"""The guard on the one-way door - Lane C's carry-forward, wired and asserted.

``lock.py`` had both primitives and tests for both, and neither was reachable: ``cli.py``
belonged to another lane and had only a ``gen`` subcommand. **Nothing else guards the test
split.** There is no second check anywhere in the repo, so if this file is green and
``cli._guard`` is called on every scoring path, the claim holds; if it is not, the claim is
that somebody remembered.

What is asserted here:

1. ``require_unlocked_or_refuse`` accepts the literal ``"1"`` and refuses every plausible
   near-miss - ``"true"``, ``"yes"``, ``"TRUE"``, ``"0"``, unset. A guard with several
   spellings is a guard someone trips over by accident.
2. ``rakshak eval --split test`` refuses **before it reads anything**, which is checked by
   running it where no dataset exists at all: the failure is the lock, not the missing file.
3. ``rakshak train`` goes through the same door. Training reads features and labels; it is
   a scoring path in every sense that matters.
4. A changed eval module is a hard fail on both paths, so the guard is two checks and not
   one wearing a hat.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rakshak import cli
from rakshak.eval.lock import LockMismatchError, SplitLockedError, require_unlocked_or_refuse

runner = CliRunner()

#: Every one of these refuses. Only the literal "1" does not.
NEAR_MISSES = ("true", "TRUE", "True", "yes", "y", "0", "", "1 ", " 1")


@pytest.mark.parametrize("value", NEAR_MISSES)
def test_only_the_literal_one_opens_the_test_split(value: str) -> None:
    with pytest.raises(SplitLockedError):
        require_unlocked_or_refuse("test", env={"RAKSHAK_UNLOCK": value})


def test_the_variable_absent_refuses() -> None:
    with pytest.raises(SplitLockedError):
        require_unlocked_or_refuse("test", env={})


def test_the_literal_one_is_accepted_and_train_and_val_are_never_gated() -> None:
    require_unlocked_or_refuse("test", env={"RAKSHAK_UNLOCK": "1"})
    require_unlocked_or_refuse("train", env={})
    require_unlocked_or_refuse("val", env={})


# ── the wiring, which is the part that did not exist ─────────────────────────


def _without_unlock() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("RAKSHAK_UNLOCK", None)
    return env


def test_eval_refuses_the_test_split_before_it_opens_anything(tmp_path: Path) -> None:
    """Pointed at an empty directory. If the refusal were downstream of the panel read,
    this would fail with a missing file instead - so the ordering is the assertion."""
    result = runner.invoke(
        cli.app,
        ["eval", "--rung", "1", "--split", "test", "--root", str(tmp_path),
         "--panel", str(tmp_path / "features.parquet")],
        env=_without_unlock(),
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SplitLockedError), result.exception


def test_train_goes_through_the_same_door(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``train`` calls ``verify_lock`` too, and a changed harness stops it.

    The tmp root holds a lock file and no ``src/rakshak/eval/`` at all, which hashes as the
    literal ``<absent>`` marker rather than being skipped - "the file that computed savings
    is gone" must not verify clean.
    """
    lock = json.loads((cli.ROOT / "EVAL-LOCK.json").read_text(encoding="utf-8"))
    (tmp_path / "EVAL-LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    result = runner.invoke(cli.app, ["train", "--rung", "2"], env=_without_unlock())
    assert result.exit_code != 0
    assert isinstance(result.exception, LockMismatchError), result.exception


def test_eval_hard_fails_on_a_changed_harness_even_on_the_validation_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lock is not only a test-split thing. Results computed against different eval
    code are not comparable to results computed against this one, on any split."""
    lock = json.loads((cli.ROOT / "EVAL-LOCK.json").read_text(encoding="utf-8"))
    (tmp_path / "EVAL-LOCK.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    result = runner.invoke(cli.app, ["eval", "--rung", "1"], env=_without_unlock())
    assert result.exit_code != 0
    assert isinstance(result.exception, LockMismatchError), result.exception


def test_the_real_lock_still_verifies_and_the_door_has_not_been_opened() -> None:
    """``eval_module_sha256`` matches and ``open_count`` is 0. If either moves without a
    T-151 entry in ``open_log``, something opened the test split."""
    from rakshak.eval.lock import read_open_count, verify_lock

    drift = verify_lock(cli.ROOT)
    assert {d.key for d in drift} <= {"generator_module_sha256", "scenario_config_sha256"}
    assert read_open_count(cli.ROOT) == 0
