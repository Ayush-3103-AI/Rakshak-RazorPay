"""The EVAL-LOCK protocol (10-eval-harness-spec.md §6; FR-025; charter §6).

The claim this module has to make checkable by someone who does not trust us:

    "The test split was opened once, after every model was final, against a harness whose
    code and configuration were hashed before any model existed."

Three mechanisms, and none of them is a matter of discipline:

1. **Hash verification.** ``EVAL-LOCK.json`` records the sha256 of every module that
   computes a number. If one changes, every eval refuses to run — results against a
   different harness are not comparable to results against this one, and the harness says
   so rather than quietly producing a number that looks like the old one.
2. **The test-split guard.** ``--split test`` refuses without ``RAKSHAK_UNLOCK=1``.
3. **The open counter.** Committed, incremented on every authorised test run, and printed
   in every results table. A counter sitting at 1 in git history is a claim anyone can
   verify from the repository alone; that verifiability is worth more than the number.

**Line endings are normalised before hashing.** ``core.autocrlf`` is ``true`` on the
Windows machine that builds this and false on Linux CI, so the same committed file has
different bytes on disk in the two places. Hashing raw bytes would make the lock fail on
CI for a reason that has nothing to do with the harness — the same class of
OS-dependent-invisible bug that cost two hours in T-101.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from rakshak.schemas import Split

if TYPE_CHECKING:
    # Annotation only. write_lock imports the concrete default lazily inside the function,
    # as it always has, so the module-level import graph is unchanged.
    from rakshak.eval.splits import SplitBoundaries

__all__ = [
    "ENFORCED_KEYS",
    "EVAL_MODULES",
    "LOCK_GLOB",
    "LOCK_PATH",
    "BrokenLockChainError",
    "LockMismatchError",
    "SplitLockedError",
    "hash_paths",
    "load_lock",
    "read_open_count",
    "record_open",
    "require_unlocked_or_refuse",
    "resolve_authoritative",
    "verify_lock",
    "write_lock",
]

#: Cycle 1. It predates the ``cycle`` key and is therefore cycle 1 by definition, and it is
#: still the file ``write_lock`` defaults to — a NEW lock is never "the authoritative one".
LOCK_PATH: Final = Path("EVAL-LOCK.json")

LOCK_GLOB: Final = "EVAL-LOCK*.json"


class BrokenLockChainError(RuntimeError):
    """The supersession chain does not resolve to exactly one live lock."""


def resolve_authoritative(root: Path) -> Path:
    """The lock that is live right now, found by resolving the supersession chain.

    Every read path defaults here rather than to ``LOCK_PATH``. Hardcoding cycle 1 was
    correct while cycle 1 was the only lock and stayed correct through cycle 2 — which
    re-sealed the harness *unchanged*, so the recorded ``eval_module_sha256`` never moved.
    Cycle 3 is the first cycle in which it does move, and at that point a hardcoded
    ``EVAL-LOCK.json`` means ``verify_lock`` checks the live tree against a permanently
    stale hash and fails forever.

    A lock that nothing supersedes is authoritative. More than one, or a ``supersedes``
    pointing at a file that is not on disk, is a refusal rather than a guess: which lock is
    live decides which numbers are comparable to which, and inferring it silently is how a
    result gets reported under a freeze it was not computed against.

    **This does not weaken enforcement.** The mismatch still raises; it is now raised
    against the lock that actually governs. The audit trail is unchanged and explicit —
    each lock records its own ``supersedes``, ``pre_registration``, ``open_count`` and
    ``open_log``, so sealing a new cycle to clear drift is visible in the chain rather than
    hidden by it.

    ``artifacts/build.py::build_lock_state`` resolves the same chain for the dashboard and
    predates this function. It should import this one instead; it was mid-edit by another
    lane when this landed, so the duplication is recorded here rather than fixed silently.
    """
    paths = sorted(root.glob(LOCK_GLOB), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(
            f"no lock file matches {LOCK_GLOB} under {root}. The eval harness is frozen "
            "before any v2 model is written (Prime Directive 1); running an eval without "
            "the lock is not a degraded mode, it is the thing the lock exists to prevent."
        )

    superseded: set[str] = set()
    for path in paths:
        target = json.loads(path.read_text(encoding="utf-8")).get("supersedes")
        if target is None:
            continue
        if not (root / target).exists():
            raise BrokenLockChainError(
                f"{path.name} supersedes {target!r}, which is not on disk. The chain is "
                "broken and which lock is live cannot be stated honestly."
            )
        superseded.add(target)

    live = [p for p in paths if p.name not in superseded]
    if len(live) != 1:
        raise BrokenLockChainError(
            f"expected exactly one unsuperseded lock, found {[p.name for p in live]}. "
            "Which lock is authoritative must be stated, never inferred."
        )
    return live[0]

#: The modules that constitute the frozen harness — the code that turns a model's scores
#: into a number. ``report.py`` and ``baf_adapter.py`` are deliberately **not** here: they
#: render and they adapt, they do not compute a metric, and listing files another lane has
#: not written yet would make the lock unverifiable the moment those files land.
EVAL_MODULES: Final = (
    "src/rakshak/eval/splits.py",
    "src/rakshak/eval/metrics.py",
    "src/rakshak/eval/oracle.py",
    "src/rakshak/eval/capacity.py",
    "src/rakshak/eval/lock.py",
)

GENERATOR_GLOB: Final = "src/rakshak/generator/*.py"
SCENARIO_CONFIG: Final = "configs/scenario_v2.yaml"

#: Hash keys whose mismatch is a HARD FAIL.
#:
#: Only ``eval_module_sha256`` is enforced, and the reason is written into the lock file
#: itself rather than hidden here. Lane C froze the harness while Lane A's generator was
#: still in flight — which STATE.md calls *preferable*, because a harness frozen before the
#: generator is finished is harder to accuse of hindsight. The unavoidable consequence is
#: that the generator hash and the scenario-config hash WILL change after this freeze. They
#: are recorded as provenance and reported as drift; enforcing them would mean the lock
#: hard-fails on Lane A's next commit, which teaches everyone to pass ``--force``, and a
#: lock people routinely override is not a lock. Pass ``strict=True`` to enforce all three
#: once the generator is itself frozen (T-116).
ENFORCED_KEYS: Final = ("eval_module_sha256",)

#: Declared BEFORE any v2 model exists (charter §2, Prime Directive 5). Not adjustable
#: after results are seen.
DECLARED_ADOPTION_MARGINS: Final = {
    "relative_pr_auc": 0.10,
    "ttd_days": 3.0,
    "note": (
        "from 00-charter-v2.md §2, declared before any v2 model existed. A rung is adopted "
        "only if it beats the previous rung by >=10% relative PR-AUC OR reduces median TTD "
        "by >=3 days at equal alerts-per-analyst-day, while holding p99 scoring latency "
        "<=10 ms per merchant on one CPU core."
    ),
}


class LockMismatchError(RuntimeError):
    """An enforced hash no longer matches. Results against this code are not comparable."""


class SplitLockedError(RuntimeError):
    """Something tried to open the test split without authorisation.

    Named without a leading ``Test`` on purpose: pytest tries to collect any class whose
    name starts with ``Test`` and emits a warning on every run, and a permanent warning is
    a thing people learn to ignore.
    """


def _normalised_bytes(path: Path) -> bytes:
    """File contents with CRLF folded to LF, so the hash is the file and not the checkout."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def hash_paths(paths: list[Path], root: Path) -> str:
    """sha256 over ``(relative_path, normalised_contents)`` for a sorted file list.

    The path is hashed alongside the contents so that renaming a module — or deleting one
    and adding another with identical text — changes the hash. A missing file hashes as
    the literal marker ``<absent>`` rather than being skipped, because "the file that
    computed savings is gone" must not verify clean.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_normalised_bytes(path) if path.is_file() else b"<absent>")
        digest.update(b"\0")
    return digest.hexdigest()


def _current_hashes(root: Path) -> dict[str, str]:
    return {
        "eval_module_sha256": hash_paths([root / m for m in EVAL_MODULES], root),
        "generator_module_sha256": hash_paths(sorted(root.glob(GENERATOR_GLOB)), root),
        "scenario_config_sha256": hash_paths([root / SCENARIO_CONFIG], root),
    }


def _git_sha(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_lock(
    root: Path,
    *,
    lock_path: Path | None = None,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    cycle: int = 1,
    supersedes: str | None = None,
    pre_registration: str | None = None,
    boundaries: SplitBoundaries | None = None,
) -> dict[str, Any]:
    """Write ``EVAL-LOCK.json``. **One-way door — there is no rollback.**

    Refuses to overwrite an existing lock. Re-freezing after models exist would destroy the
    only property the file is for, so re-running this is an error rather than an update.
    """
    target = lock_path if lock_path is not None else root / LOCK_PATH
    if target.exists():
        raise FileExistsError(
            f"{target} already exists. The lock is written once, before any v2 model "
            "exists (charter §6.2). Rewriting it after the fact destroys the claim it "
            "makes; if the harness genuinely must change, that is a DESCEND and a new "
            "cycle, not an edit."
        )

    # The window the lock records must be the window that is actually scored. Taking it
    # from DEFAULT_BOUNDARIES was safe while they were the same fact, but T-0101 moved the
    # geometry to 365 days by passing explicit day tuples and deliberately NOT editing
    # eval/splits.py (see cli._boundaries' docstring), so the dataclass defaults are still
    # the 180-day window: train (0,119) / val (120,149) / test (150,179). Sealing from them
    # writes a lock whose split_boundaries contradict the pre-registration and every day the
    # harness actually scores. Caller passes the derived boundaries; the default reproduces
    # the old behaviour for cycles 1 and 2, which were written when the two agreed.
    if boundaries is None:
        from rakshak.eval.splits import DEFAULT_BOUNDARIES

        b = DEFAULT_BOUNDARIES
    else:
        b = boundaries

    lock: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        # Cycle 2 carried these three and write_lock did not emit them — they were added to
        # the file by hand. resolve_authoritative() reads `supersedes` to find the live
        # lock, so a hand-added chain link is a chain that breaks the next time someone
        # forgets. Emitted here now; default cycle 1 / no supersedes reproduces the old
        # output for any existing caller.
        "cycle": cycle,
        "supersedes": supersedes,
        "pre_registration": pre_registration,
        **_current_hashes(root),
        "eval_modules": list(EVAL_MODULES),
        "generator_glob": GENERATOR_GLOB,
        "scenario_config": SCENARIO_CONFIG,
        "enforced": list(ENFORCED_KEYS),
        "enforcement_note": (
            "Only eval_module_sha256 is a hard fail. Lane C froze the harness while Lane A's "
            "generator was still in flight, which project-context/STATE.md calls preferable: "
            "a harness frozen before the generator is finished is harder to accuse of "
            "hindsight. The generator and scenario-config hashes are therefore recorded as "
            "provenance of what existed at freeze time and reported as drift, not enforced. "
            "Enforcing them here would hard-fail on Lane A's next commit and train everyone "
            "to override the lock, and a lock that is routinely overridden is not a lock. "
            "verify_lock(strict=True) enforces all three once the generator is frozen (T-116)."
        ),
        "seeds": list(seeds),
        "split_boundaries": {"train": list(b.train), "val": list(b.val), "test": list(b.test)},
        "split_origin": b.origin.isoformat(),
        "capacity_k": 50,
        "capacity_per_n_merchants": 10000,
        "metrics": [
            "pr_auc",
            "roc_auc",
            "ece",
            "savings",
            "savings_floor_random",
            "savings_floor_all_pass",
            "savings_floor_all_hold",
            "savings_floor_volume_rank",
            "precision_at_k",
            "recall_at_k",
            "alerts_per_day",
            "ttd_median_days",
            "detection_rate_d7",
            "detection_rate_d14",
            "detection_rate_d30",
            "gap_to_oracle",
            "alert_jaccard_wow",
            "recall_by_typology",
            "p99_latency_ms",
            "state_bytes_p99",
            "model_size_mb",
            # Cycle-3 additions, declared in docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md §2
            # before any of them was implemented and before any Rung 5-8 code exists.
            # Additive: nothing above is removed or redefined (§3), and the three are
            # inert for Rungs 0-4, which stay judged on EVAL-LOCK-CYCLE2.json.
            #
            # They are listed HERE, in the module, rather than typed into the lock file,
            # because eval_module_sha256 covers lock.py: adding them to the JSON after the
            # cycle-3 lock is sealed would need an edit to this file, which would break the
            # very hash the lock had just recorded. That is the ordering trap §4 exists to
            # avoid, and it is why T-0118 lands before the lock is written.
            "false_hold_coverage",
            "onset_localisation_error",
            "tpp_rescaled_ks",
        ],
        "cost_asymmetry_ratios": [0.01, 0.1, 1.0, 10.0, 100.0],
        "declared_adoption_margins": DECLARED_ADOPTION_MARGINS,
        "frozen_at_git_sha": _git_sha(root),
        "open_count": 0,
        "open_log": [],
    }
    target.write_text(json.dumps(lock, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return lock


def load_lock(root: Path, *, lock_path: Path | None = None) -> dict[str, Any]:
    target = lock_path if lock_path is not None else resolve_authoritative(root)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. The eval harness is frozen before any v2 model is "
            "written (Prime Directive 1); running an eval without the lock is not a "
            "degraded mode, it is the thing the lock exists to prevent."
        )
    parsed: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
    return parsed


@dataclass(frozen=True, slots=True)
class Drift:
    """A recorded-but-unenforced hash that no longer matches. Reported, never silent."""

    key: str
    expected: str
    actual: str


def verify_lock(
    root: Path, *, lock_path: Path | None = None, strict: bool = False
) -> list[Drift]:
    """Recompute the hashes. Enforced mismatch -> hard fail; the rest come back as drift.

    ``strict=True`` promotes every recorded hash to enforced — use it once the generator
    is itself frozen (T-116).
    """
    target = lock_path if lock_path is not None else resolve_authoritative(root)
    lock = load_lock(root, lock_path=target)
    current = _current_hashes(root)
    enforced = set(lock.get("enforced", ENFORCED_KEYS)) if not strict else set(current)

    drift: list[Drift] = []
    for key, actual in current.items():
        expected = lock.get(key)
        if expected == actual:
            continue
        if key in enforced:
            raise LockMismatchError(
                # Name the lock that actually governs. Saying "EVAL-LOCK.json" once there
                # is more than one lock sends the reader to the wrong file.
                f"{key} does not match {target.name}: expected {expected}, got {actual}. "
                "The eval code changed after the lock was written, so results computed "
                "against it are not comparable to anything measured before. Restore the "
                "frozen modules, or accept that this is a new harness and say so."
            )
        drift.append(Drift(key=key, expected=str(expected), actual=actual))
    return drift


def require_unlocked_or_refuse(split: Split, *, env: dict[str, str] | None = None) -> None:
    """Refuse the test split unless ``RAKSHAK_UNLOCK=1`` (FR-025).

    The variable is not a convenience. Setting it is the single, deliberate, auditable act
    that opens the test split, and it happens exactly once, in T-151, after every rung is
    final. If you are reaching for it to debug a model, debug on the validation split —
    you are about to destroy the most valuable property of this project.
    """
    if split != "test":
        return
    environ = os.environ if env is None else env
    if environ.get("RAKSHAK_UNLOCK") != "1":
        raise SplitLockedError(
            "refusing to open the test split. EVAL-LOCK.json records that it is opened "
            "exactly once, at the end (charter §6.3, Prime Directive 1). Set "
            "RAKSHAK_UNLOCK=1 only in T-151, after every rung is final; the open counter "
            "in EVAL-LOCK.json is committed and anyone can read it in the git history."
        )


def read_open_count(root: Path, *, lock_path: Path | None = None) -> int:
    return int(load_lock(root, lock_path=lock_path)["open_count"])


def record_open(
    root: Path, rungs_scored: list[int], *, lock_path: Path | None = None
) -> dict[str, Any]:
    """Append to ``open_log``, increment ``open_count``, write, and say to commit it."""
    target = lock_path if lock_path is not None else resolve_authoritative(root)
    lock = load_lock(root, lock_path=lock_path)
    lock["open_log"].append(
        {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "git_sha": _git_sha(root),
            "rungs_scored": list(rungs_scored),
        }
    )
    lock["open_count"] = int(lock["open_count"]) + 1
    target.write_text(json.dumps(lock, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return lock
