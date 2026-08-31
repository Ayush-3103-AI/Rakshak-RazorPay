"""T-133 — the EVAL-LOCK protocol.

Every test here runs against a synthetic repo root in ``tmp_path``. Nothing in this file
touches the real ``EVAL-LOCK.json``: the real one is a one-way door and a test that can
rewrite it is a rollback in disguise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rakshak.eval.lock import (
    EVAL_MODULES,
    LockMismatchError,
    SplitLockedError,
    hash_paths,
    load_lock,
    read_open_count,
    record_open,
    require_unlocked_or_refuse,
    verify_lock,
    write_lock,
)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A repo-shaped tree with stand-in eval, generator and config files."""
    for rel in EVAL_MODULES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# stand-in for {rel}\nVALUE = 1\n", encoding="utf-8")
    gen = tmp_path / "src" / "rakshak" / "generator"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "engine.py").write_text("# generator\n", encoding="utf-8")
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "scenario_v2.yaml").write_text("seed: 42\n", encoding="utf-8")
    return tmp_path


# ─────────────────────────── writing the lock ───────────────────────────


def test_a_fresh_lock_has_open_count_zero_and_an_empty_log() -> None:
    """The whole point of the counter is that it starts at zero and is committed there."""
    # Uses the real repository lock, read-only.
    root = Path(__file__).resolve().parents[2]
    lock = load_lock(root)
    assert lock["open_count"] == 0
    assert lock["open_log"] == []


def test_write_lock_produces_every_field_the_spec_names(fake_root: Path) -> None:
    lock = write_lock(fake_root)
    for key in (
        "created_at",
        "scenario_config_sha256",
        "eval_module_sha256",
        "generator_module_sha256",
        "seeds",
        "split_boundaries",
        "metrics",
        "declared_adoption_margins",
        "open_count",
        "open_log",
    ):
        assert key in lock, key
    assert lock["open_count"] == 0
    assert lock["open_log"] == []
    assert lock["split_boundaries"] == {"train": [0, 119], "val": [120, 149], "test": [150, 179]}
    assert lock["declared_adoption_margins"]["relative_pr_auc"] == 0.10
    assert lock["declared_adoption_margins"]["ttd_days"] == 3.0


def test_the_lock_is_written_once_and_refuses_to_be_rewritten(fake_root: Path) -> None:
    """One-way door. Re-freezing after models exist destroys the claim the file makes."""
    write_lock(fake_root)
    with pytest.raises(FileExistsError, match="written once"):
        write_lock(fake_root)


def test_the_declared_margins_predate_any_model(fake_root: Path) -> None:
    """Prime Directive 5: the margins are declared before the run, not adjusted after."""
    lock = write_lock(fake_root)
    note = lock["declared_adoption_margins"]["note"]
    assert "before any v2 model existed" in note


# ─────────────────────────── hash verification ───────────────────────────


def test_an_unmodified_tree_verifies_clean(fake_root: Path) -> None:
    write_lock(fake_root)
    assert verify_lock(fake_root) == []


def test_a_modified_eval_module_causes_a_hash_mismatch_hard_fail(fake_root: Path) -> None:
    """T-133's done-when. Not a warning, not a drift entry — a refusal to run."""
    write_lock(fake_root)
    victim = fake_root / EVAL_MODULES[1]  # metrics.py
    victim.write_text(victim.read_text(encoding="utf-8") + "\nVALUE = 2\n", encoding="utf-8")
    with pytest.raises(LockMismatchError, match="eval_module_sha256"):
        verify_lock(fake_root)


def test_a_deleted_eval_module_also_hard_fails(fake_root: Path) -> None:
    """"The file that computed savings is gone" must not verify clean."""
    write_lock(fake_root)
    (fake_root / EVAL_MODULES[2]).unlink()
    with pytest.raises(LockMismatchError):
        verify_lock(fake_root)


def test_a_renamed_eval_module_hard_fails_even_with_identical_text(fake_root: Path) -> None:
    write_lock(fake_root)
    victim = fake_root / EVAL_MODULES[3]
    victim.rename(victim.with_name("renamed.py"))
    with pytest.raises(LockMismatchError):
        verify_lock(fake_root)


def test_generator_drift_is_reported_not_enforced(fake_root: Path) -> None:
    """Lane C froze before Lane A finished. Generator drift is expected, recorded, and
    surfaced — and it must never be silent."""
    write_lock(fake_root)
    (fake_root / "src" / "rakshak" / "generator" / "engine.py").write_text(
        "# generator, changed\n", encoding="utf-8"
    )
    drift = verify_lock(fake_root)
    assert [d.key for d in drift] == ["generator_module_sha256"]
    assert drift[0].expected != drift[0].actual


def test_strict_mode_promotes_every_recorded_hash_to_enforced(fake_root: Path) -> None:
    write_lock(fake_root)
    (fake_root / "configs" / "scenario_v2.yaml").write_text("seed: 43\n", encoding="utf-8")
    assert [d.key for d in verify_lock(fake_root)] == ["scenario_config_sha256"]
    with pytest.raises(LockMismatchError, match="scenario_config_sha256"):
        verify_lock(fake_root, strict=True)


def test_the_lock_records_exactly_which_files_it_covers(fake_root: Path) -> None:
    """A lock nobody can enumerate is a lock nobody can check."""
    lock = write_lock(fake_root)
    assert lock["eval_modules"] == list(EVAL_MODULES)
    assert lock["scenario_config"] == "configs/scenario_v2.yaml"
    assert lock["enforced"] == ["eval_module_sha256"]
    assert "harder to accuse of hindsight" in lock["enforcement_note"]


def test_a_missing_lock_is_an_error_not_a_degraded_mode(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="frozen before any v2 model"):
        load_lock(tmp_path)


# ─────────────────────────── line endings ───────────────────────────


def test_hashing_is_immune_to_crlf(tmp_path: Path) -> None:
    """core.autocrlf is true on the build machine and false on Linux CI, so the same
    committed file has different bytes in the two places. Hashing raw bytes would fail CI
    for a reason with nothing to do with the harness — the T-101 class of bug."""
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    (lf / "m.py").write_bytes(b"a = 1\nb = 2\n")
    (crlf / "m.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    assert hash_paths([lf / "m.py"], lf) == hash_paths([crlf / "m.py"], crlf)


def test_hashing_is_not_immune_to_content(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_bytes(b"a = 1\n")
    first = hash_paths([tmp_path / "m.py"], tmp_path)
    (tmp_path / "m.py").write_bytes(b"a = 2\n")
    assert hash_paths([tmp_path / "m.py"], tmp_path) != first


# ─────────────────────────── the test-split guard ───────────────────────────


def test_the_test_split_is_refused_without_the_unlock_variable() -> None:
    with pytest.raises(SplitLockedError, match="EVAL-LOCK.json"):
        require_unlocked_or_refuse("test", env={})


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE"])
def test_only_the_exact_value_one_unlocks(value: str) -> None:
    """A guard that accepts "true" is a guard someone opens by accident."""
    with pytest.raises(SplitLockedError):
        require_unlocked_or_refuse("test", env={"RAKSHAK_UNLOCK": value})


def test_the_unlock_variable_opens_the_test_split() -> None:
    require_unlocked_or_refuse("test", env={"RAKSHAK_UNLOCK": "1"})


@pytest.mark.parametrize("split", ["train", "val"])
def test_train_and_val_never_need_the_unlock(split: str) -> None:
    require_unlocked_or_refuse(split, env={})  # type: ignore[arg-type]


# ─────────────────────────── the open counter ───────────────────────────


def test_recording_an_open_increments_the_counter_and_logs_it(fake_root: Path) -> None:
    write_lock(fake_root)
    assert read_open_count(fake_root) == 0
    lock = record_open(fake_root, [0, 1, 2, 3])
    assert lock["open_count"] == 1
    assert len(lock["open_log"]) == 1
    entry = lock["open_log"][0]
    assert entry["rungs_scored"] == [0, 1, 2, 3]
    assert set(entry) == {"timestamp", "git_sha", "rungs_scored"}
    assert read_open_count(fake_root) == 1


def test_the_counter_persists_to_disk(fake_root: Path) -> None:
    write_lock(fake_root)
    record_open(fake_root, [2])
    record_open(fake_root, [3])
    on_disk = json.loads((fake_root / "EVAL-LOCK.json").read_text(encoding="utf-8"))
    assert on_disk["open_count"] == 2
    assert len(on_disk["open_log"]) == 2


def test_recording_an_open_does_not_touch_the_hashes(fake_root: Path) -> None:
    """Incrementing the counter must not silently re-freeze the harness."""
    before = write_lock(fake_root)["eval_module_sha256"]
    record_open(fake_root, [1])
    assert load_lock(fake_root)["eval_module_sha256"] == before


# ─────────────────────────── the real lock ───────────────────────────


def test_the_committed_lock_verifies_against_this_working_tree() -> None:
    """If this fails, an eval module changed after the freeze and no result is comparable."""
    root = Path(__file__).resolve().parents[2]
    drift = verify_lock(root)
    # Generator/config drift is expected and reported; eval drift would have raised above.
    assert all(d.key != "eval_module_sha256" for d in drift)
