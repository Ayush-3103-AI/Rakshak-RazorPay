"""The artefact generator — `make artifacts` (T-0126, #60).

Reads committed results and lock files, emits validated JSON into ``artifacts/``. It is a
**pure function of its input files' contents**: no clock, no git subprocess, no network,
no ``data/`` writes. That is not fastidiousness — "regenerating from the same committed
results is byte-identical" is an acceptance criterion, and the two cheapest ways to break
it are a ``generated_at`` stamp and a ``git rev-parse`` that moves when someone else
commits. Every commit SHA reported here is read out of a lock file or a result row, where
it was recorded at the time the number was computed.

**A missing input is a named absence, not a silent one.** If the G5 gate has not written
its series, the manifest carries ``status: "MISSING"`` with the reason and the path that
was looked for, and no ``g5_confounder_null.json`` is written. The loader then renders a
named error rather than an empty-but-plausible chart, which is the same contract one step
earlier. Fabricating a placeholder would be the one failure mode this whole ticket exists
to prevent.

**The test split is shut until the counter says otherwise.** A TEST-split result row is
refused while the authoritative lock's ``open_count`` is 0 — the split opens exactly once,
in T-0116, after every rung is final, and until that has happened a TEST row is either a
mislabelled file or a leak. Neither belongs on a judge-facing page. After the counter
moves, the same row is emitted normally: the guard is the lock, not a taboo.

**The ladder does not borrow the live lock's authority.** Its provenance carries the
result rows' own ``eval_lock_sha`` and ``git_sha`` beside the authoritative lock's, and it
says which *cycle* froze the harness each row was scored under. A row scored under a
superseded cycle is not drift — the cycle-3 pre-registration commits in writing that
Rungs 0-4 stay judged on cycle 2 and are not rescored — so it is recorded as such and
rendered as such, never silently promoted to current. A row matching **no** lock in the
chain is a refusal. What the rows cannot certify — the generator and scenario-config
hashes, which ``EvalResult`` does not record — is said in ``harness_note`` rather than
left for the reader to assume.

**Lock files are discovered, never enumerated, and by one implementation.** ``EVAL-LOCK.json``
is cycle 1 and ``EVAL-LOCK-CYCLE2.json`` supersedes it; a further cycle is expected and must
need no code change here, so nothing names a lock file. Which lock is live comes from
``eval.lock.resolve_authoritative`` — the same function ``verify_lock`` and the one-way door
obey — rather than from a second resolver here, because two answers to "which lock governs"
is how the dashboard ends up citing one lock while the harness enforces another. This module
adds only what the dashboard needs on top: the per-lock detail and the back-links, with
``authoritative_lock`` as an explicit field so nothing has to infer it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, Final

import yaml

from rakshak.artifacts import (
    ARTIFACTS_DIR,
    RUNG_STATUS_VALUES,
    SCHEMA_VERSION,
    ArtifactSchemaError,
    canonical_bytes,
    envelope,
    sanitise,
    sha256_bytes,
    split_label,
    validate,
)
from rakshak.eval.lock import LOCK_GLOB, BrokenLockChainError, resolve_authoritative
from rakshak.schemas import EvalResult

__all__ = [
    "build_all",
    "build_cost_sweep",
    "build_g5",
    "build_journey",
    "build_ladder",
    "build_lock_state",
    "build_rung_roster",
    "main",
]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]

DEFAULT_RESULTS_DIR: Final = Path("data/v2/eval")
#: Where the G5 gate is expected to drop its series. Nothing in this module writes it —
#: `tests/gates/test_g5_confounder_null.py` owns the measurement and must own the dump.
DEFAULT_G5_PATH: Final = Path("data/v2/gates/g5_series.json")
#: Hand-maintained, committed, and reviewed by the lead. The ladder can only show rungs
#: that were scored, so a cut or deferred rung is invisible to it; this is where one is
#: named as cut (#64).
DEFAULT_ROSTER_PATH: Final = Path("configs/rung_roster.yaml")
#: Written by `scripts/cost_sweep.py`, which owns the measurement. Nothing here computes
#: a savings number; this module reshapes a committed dump into chart-ready series.
DEFAULT_SWEEP_PATH: Final = Path("docs/results/cost_sweep.json")
#: Hand-maintained and committed, exactly like the roster. Prime Directive 2 makes prior
#: generations' numbers immutable, so they are literals with citations rather than a
#: computation — and a literal nobody can trace is an assertion, so the builder refuses one.
DEFAULT_JOURNEY_PATH: Final = Path("configs/journey.yaml")

#: The public generation labels. The charter's own vocabulary ("v1", "v2") means something
#: different in the sibling repo, where "v1" is the constrained-RL tree; these three names
#: collide with nothing and the mapping travels on the artefact so the two can be reconciled.
GENERATION_IDS: Final[tuple[str, ...]] = ("G1", "G2", "G3")

LOCK_HASH_KEYS: Final = (
    "eval_module_sha256",
    "generator_module_sha256",
    "scenario_config_sha256",
)

#: Every float field on ``EvalResult``, derived from the dataclass so that a metric added
#: upstream reaches the dashboard without this file being edited.
_NON_METRIC: Final = frozenset(
    {
        "rung",
        "split",
        "recall_by_typology",
        "floor_fail",
        "eval_lock_sha",
        "git_sha",
        "open_count",
        # `label` is the policy's name, not a measurement. METRIC_KEYS is derived from
        # EvalResult's fields precisely so a metric added upstream reaches the dashboard
        # without editing this file — which means a non-metric added upstream must be
        # excluded here, or the ladder publishes a string in a float column.
        "label",
    }
)
METRIC_KEYS: Final[tuple[str, ...]] = tuple(
    f.name for f in fields(EvalResult) if f.name not in _NON_METRIC and f.name != "cost_scenario"
)
#: Extra keys `cli.py` writes beside the row that the ladder needs: the oracle ceiling the
#: gap is measured against, and the capacity the whole thing is budgeted under.
EXTRA_KEYS: Final[tuple[str, ...]] = (
    "oracle_savings",
    "capacity_k",
    "n_merchants_scored",
    "n_rows_kept",
    "n_censored_dropped",
    "n_features",
)


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_result_rows(directory: Path) -> list[dict[str, Any]]:
    """Every ``*.json`` result the scoring path wrote, sorted by filename.

    Validated against ``EvalResult``'s field list rather than trusted: a file missing
    ``prevalence`` is not a result row, and rendering it would reproduce exactly the v1
    failure (a PR-AUC printed without the prevalence it was measured at, FR-021).
    """
    required = {f.name for f in fields(EvalResult)} - {"cost_scenario", "floor_fail"}
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.name):
        record = _read_json(path)
        missing = required - set(record)
        if missing:
            raise ArtifactSchemaError(
                "ladder", f"{path.name} is not an EvalResult: missing {sorted(missing)}"
            )
        record["_source"] = path.name
        record["_seed"] = _seed_of(path.name)
        rows.append(record)
    return rows


def _seed_of(filename: str) -> int | None:
    stem = filename.removesuffix(".json")
    tail = stem.rsplit("_seed", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def _rel(path: Path, root: Path) -> str:
    """Repo-relative posix path, so provenance does not record whose laptop built it.

    Falls back to the absolute path only when the input genuinely lives outside the tree,
    which is a test or a one-off override and never the committed path.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _inputs_provenance(paths: Sequence[Path], root: Path) -> list[dict[str, str]]:
    """Path + content hash for every file that fed an artefact. Sorted, never walk order.

    Line endings are normalised before hashing, for the reason ``eval/lock.py`` gives:
    ``core.autocrlf`` differs between the Windows build machine and Linux CI, so a hash of
    the raw bytes would be a hash of the checkout rather than of the file.
    """
    return [
        {
            "path": _rel(p, root),
            "sha256": sha256_bytes(p.read_bytes().replace(b"\r\n", b"\n")),
        }
        for p in sorted(paths, key=lambda q: _rel(q, root))
    ]


def _results_provenance(
    rows: Sequence[dict[str, Any]], locks: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """What the *rows* say about themselves, placed in the lock chain that governs them.

    Without this the ladder carries the authoritative lock's shas and nothing else, which
    reads as "these numbers were computed under this lock" — and that is exactly the claim
    the artefact is not entitled to make. A result row records the harness hash and the
    commit it ran at; it does not record the generator or scenario-config hash, so it
    cannot certify the *geometry* the population was drawn at. The dashboard gets both
    facts and the gap between them, stated rather than papered over.

    **The harness check is over the whole supersession chain, not only the live lock.**
    A row scored under a superseded cycle is not drift: it is the pre-registered state.
    ``docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md`` §3 — *"No existing rung is rescored and
    no committed number moves. Rungs 0-4 are judged on the cycle-2 lock exactly as
    before"* — is a commitment that the rungs already scored STAY on cycle 2 once cycle 3
    seals. Comparing every row against the live lock alone would turn that commitment into
    a refusal and stop the ladder emitting on the day the next lock lands, which is both
    wrong and the opposite of what the lock is for.

    So a row is legitimate if its ``eval_lock_sha`` matches **some** lock in the resolved
    chain, and the artefact then records *which* — ``results_scored_under`` maps each
    row-sha to the cycles and lock files that froze that harness, so the dashboard can
    render "scored under cycle 2" beside a cycle-3 banner instead of implying the number
    is current. ``results_are_current`` is that comparison, pre-computed and named.

    A row whose sha matches **no** lock in the chain is still a hard refusal. That one is
    genuine drift — a number computed against a harness this project never froze — and it
    is not comparable to anything measured before, which is the whole reason the lock
    exists. Cycles 1 and 2 record the *same* ``eval_module_sha256`` (cycle 2 re-sealed the
    harness unchanged), so a sha can legitimately name more than one lock; both are listed
    rather than one being picked, because picking would be a guess.
    """
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for lock in locks:
        by_sha.setdefault(str(lock["hashes"]["eval_module_sha256"]), []).append(lock)

    live = next(lock for lock in locks if lock["authoritative"])
    authoritative = str(live["hashes"]["eval_module_sha256"])

    scored_under: dict[str, dict[str, Any]] = {}
    strays: dict[str, list[str]] = {}
    for row in rows:
        sha = str(row["eval_lock_sha"])
        source = str(row.get("_source", row.get("label", "<row>")))
        matched = by_sha.get(sha)
        if matched is None:
            strays.setdefault(sha, []).append(source)
            continue
        entry = scored_under.setdefault(
            sha,
            {
                "cycles": sorted({int(lock["cycle"]) for lock in matched}),
                "locks": sorted(str(lock["file"]) for lock in matched),
                "is_authoritative_lock": sha == authoritative,
                "sources": [],
            },
        )
        entry["sources"].append(source)
    for entry in scored_under.values():
        entry["sources"] = sorted(set(entry["sources"]))

    if strays:
        raise ArtifactSchemaError(
            "ladder",
            f"result rows {sorted(s for v in strays.values() for s in v)} were computed "
            f"against eval module {sorted(strays)}, which matches no lock in the "
            f"supersession chain {[str(lock['file']) for lock in locks]} (authoritative "
            f"lock: {live['file']}). A superseded cycle would be fine and is recorded as "
            "such; a harness this project never froze is not, because nothing bounds what "
            "it computed. Re-run `make eval` against a lock in the chain.",
        )

    return {
        "results_eval_lock_sha": sorted(scored_under),
        "results_git_sha": sorted({str(r["git_sha"]) for r in rows}),
        "results_open_count": sorted({int(r["open_count"]) for r in rows}),
        "authoritative_cycle": int(live["cycle"]),
        "results_scored_under": scored_under,
        "results_are_current": all(
            entry["is_authoritative_lock"] for entry in scored_under.values()
        ),
        "harness_note": (
            "Every results_eval_lock_sha is verified to match a lock in the supersession "
            "chain, and results_scored_under names which cycle froze each one. "
            "results_are_current is false when any row was scored under a SUPERSEDED "
            "cycle — that is the pre-registered state for a rung the newer cycle did not "
            "rescore, not drift, and such a row must be rendered as 'scored under cycle "
            "N', never as a current number. EvalResult rows do NOT record "
            "generator_module_sha256 or scenario_config_sha256, so this artefact cannot "
            "certify that they were computed at any lock's geometry. Compare "
            "results_git_sha with that cycle's frozen_at_git_sha before presenting a "
            "number as current."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# lock_state — N locks, discovered, with the supersession chain resolved
# ─────────────────────────────────────────────────────────────────────────────


def build_lock_state(root: Path) -> tuple[dict[str, Any], list[Path]]:
    """The lock artefact: three hashes, open counter and freezing commit, per lock.

    ``EVAL-LOCK.json`` predates the ``cycle`` key and is therefore cycle 1 by definition.
    Everything after it declares its own cycle and what it supersedes. A lock that nothing
    supersedes is authoritative; more than one of those means the chain is broken and the
    dashboard would have to guess, so it is a refusal instead.
    """
    paths = sorted(root.glob(LOCK_GLOB), key=lambda p: p.name)
    if not paths:
        raise ArtifactSchemaError("lock_state", f"no lock file matches {LOCK_GLOB} under {root}")

    entries: list[dict[str, Any]] = []
    for path in paths:
        lock = _read_json(path)
        entries.append(
            {
                "file": path.name,
                "cycle": int(lock.get("cycle", 1)),
                "supersedes": lock.get("supersedes"),
                "superseded_by": None,
                "authoritative": False,
                # From the lock file, not from the clock: this is when the freeze happened.
                "created_at": lock.get("created_at"),
                "frozen_at_git_sha": lock.get("frozen_at_git_sha"),
                "hashes": {key: lock.get(key) for key in LOCK_HASH_KEYS},
                "enforced": sorted(lock.get("enforced", [])),
                "open_count": int(lock["open_count"]),
                "open_log": lock.get("open_log", []),
                "seeds": list(lock.get("seeds", [])),
                "capacity_k": lock.get("capacity_k"),
                "split_boundaries": lock.get("split_boundaries", {}),
                "pre_registration": lock.get("pre_registration"),
            }
        )
    entries.sort(key=lambda e: (e["cycle"], e["file"]))

    by_file = {e["file"]: e for e in entries}
    for entry in entries:
        target = entry["supersedes"]
        if target is None:
            continue
        if target not in by_file:
            raise ArtifactSchemaError(
                "lock_state",
                f"{entry['file']} supersedes {target!r}, which is not on disk; the chain is "
                "broken and which lock is live cannot be stated honestly",
            )
        by_file[target]["superseded_by"] = entry["file"]

    # Which lock is live is decided by `eval/lock.py`, not a second time here. That
    # function is what `verify_lock` and the one-way door already obey, and two
    # implementations of "which lock is authoritative" is how they drift apart — at which
    # point the dashboard cites one lock and the harness enforces another.
    try:
        live_path = resolve_authoritative(root)
    except BrokenLockChainError as exc:
        raise ArtifactSchemaError("lock_state", str(exc)) from exc
    live = by_file[live_path.name]
    live["authoritative"] = True
    if live["cycle"] != max(e["cycle"] for e in entries):
        raise ArtifactSchemaError(
            "lock_state",
            f"{live['file']} is unsuperseded but is not the highest cycle; the "
            "supersession chain and the cycle numbers disagree",
        )

    payload = {
        "authoritative_lock": live["file"],
        "n_locks": len(entries),
        "test_split_opened": sum(int(e["open_count"]) for e in entries) > 0,
        "locks": entries,
    }
    return payload, paths


# ─────────────────────────────────────────────────────────────────────────────
# ladder — rungs, floors and the oracle gap, aggregated over seeds
# ─────────────────────────────────────────────────────────────────────────────


def _median(values: Sequence[float]) -> tuple[float | None, dict[str, int]]:
    """Median of the finite values, plus a census of the non-finite ones.

    A rung whose TTD is ``+inf`` on all five seeds has "never detected" as its result, and
    that is what ``{"Infinity": 5}`` beside a ``null`` says. Dropping the non-finite values
    into the median instead would produce a plausible number for a thing that never
    happened.
    """
    finite: list[float] = []
    census: dict[str, int] = {}
    for value in values:
        number = float(value)
        if math.isfinite(number):
            finite.append(number)
            continue
        token = "NaN" if math.isnan(number) else ("Infinity" if number > 0 else "-Infinity")
        census[token] = census.get(token, 0) + 1
    return (statistics.median(finite) if finite else None), census


def build_ladder(
    rows: Sequence[dict[str, Any]], *, test_split_opened: bool = False
) -> dict[str, Any]:
    """One row per ``(rung, label, split, cost_scenario)``, aggregated across seeds.

    Floors are columns rather than a separate artefact, because a savings number without
    the four floors beside it is the exact shape of the v2 finding that random selection
    won at inflated prevalence. ``gap_to_oracle`` and ``oracle_savings`` travel with them
    for the same reason: an unanchored absolute is not a result.

    ``test_split_opened`` is the lock's open counter, not a caller's opinion. A TEST row
    that exists while the counter is still zero was produced without an authorised open —
    it is either a mislabelled file or a leak — and either way publishing it would put a
    test-split number on a judge-facing page before T-0116 opened the split. Refuse.
    """
    if not rows:
        raise ArtifactSchemaError(
            "ladder",
            "no EvalResult rows to render. Artefacts report what was measured; they do not "
            "invent a table. If you expected TEST rows, they do not exist yet.",
        )

    if not test_split_opened:
        leaked = sorted(
            str(r.get("_source", r.get("label", "<row>")))
            for r in rows
            if split_label(str(r["split"])) == "TEST"
        )
        if leaked:
            raise ArtifactSchemaError(
                "ladder",
                f"TEST-split rows {leaked} while the authoritative lock's open counter is 0. "
                "The test split opens exactly once, in T-0116, after every rung is final; "
                "until the counter says so no test number may be published.",
            )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["rung"]),
            str(row.get("label", f"rung{row['rung']}")),
            split_label(str(row["split"])),
            str(row.get("cost_scenario", "base")),
        )
        groups.setdefault(key, []).append(row)

    rungs: list[dict[str, Any]] = []
    for (rung, label, split, scenario), members in sorted(groups.items(), key=lambda kv: kv[0]):
        metrics: dict[str, float | None] = {}
        non_finite: dict[str, dict[str, int]] = {}
        for metric in METRIC_KEYS + EXTRA_KEYS:
            present = [m[metric] for m in members if m.get(metric) is not None]
            if not present:
                continue
            value, census = _median(present)
            metrics[metric] = value
            if census:
                non_finite[metric] = census

        typologies = sorted({t for m in members for t in (m.get("recall_by_typology") or {})})
        recall_by_typology: dict[str, float | None] = {}
        for typology in typologies:
            value, census = _median(
                [
                    m["recall_by_typology"][typology]
                    for m in members
                    if typology in m["recall_by_typology"]
                ]
            )
            recall_by_typology[typology] = value
            if census:
                non_finite.setdefault("recall_by_typology", {})
                for token, count in census.items():
                    non_finite["recall_by_typology"][f"{typology}:{token}"] = count

        floor_fail = sorted({f for m in members for f in (m.get("floor_fail") or [])})
        lock_shas = sorted({str(m["eval_lock_sha"]) for m in members})
        git_shas = sorted({str(m["git_sha"]) for m in members})
        rungs.append(
            {
                "rung": rung,
                "label": label,
                "split": split,
                "cost_scenario": scenario,
                "n_seeds": len(members),
                "seeds": sorted(s for s in (m["_seed"] for m in members) if s is not None),
                "metrics": metrics,
                "recall_by_typology": recall_by_typology,
                "non_finite": non_finite,
                "floor_fail": floor_fail,
                "n_seeds_floor_fail": sum(1 for m in members if m.get("floor_fail")),
                "beats_all_floors": not floor_fail,
                # Per-row provenance. Aggregating over seeds must not aggregate over
                # harnesses, so a group whose members disagree says so rather than hiding it.
                "eval_lock_sha": lock_shas,
                "git_sha": git_shas,
                "provenance_consistent": len(lock_shas) == 1 and len(git_shas) == 1,
                "sources": sorted(str(m["_source"]) for m in members),
            }
        )

    capacities = {r["metrics"].get("capacity_k") for r in rungs}
    return {
        "rungs": rungs,
        "metric_keys": list(METRIC_KEYS + EXTRA_KEYS),
        "capacity_k": next(iter(capacities)) if len(capacities) == 1 else None,
        "splits": sorted({r["split"] for r in rungs}),
        "typologies": sorted({t for r in rungs for t in r["recall_by_typology"]}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# g5_confounder_null — the lead figure
# ─────────────────────────────────────────────────────────────────────────────


#: Which confounder windows are the adversarial ones and which are controls is a property
#: of the run, not of the reader. #62 requires the shading to be labelled from the artefact
#: rather than from a hardcoded ``["P2", "P3"]`` in a component, so ``role`` is required of
#: the gate dump and refused if absent.
G5_WINDOW_ROLES: Final[frozenset[str]] = frozenset({"adversarial", "control"})

#: The window convention, emitted as a field rather than left to the reader.
#:
#: **Half-open.** ``confounders.py`` computes ``end = min(n_days, start + duration_days)``,
#: so P1's ``duration_days: 5`` at day 93 emits ``[93, 98)`` — five days, 93 through 97.
#: Read as inclusive that is six, and P2's ``duration_hours: 6`` window ``[57, 58)`` becomes
#: a two-day band instead of the one day it is. The error message here used to say
#: "spans days [start, end]", which asserted the wrong one; #62 shades from this field.
#: The bound is ``end <= n_days``, not ``end < n_days``, for the same reason: a confounder
#: running to the last day emits ``end == n_days`` and is legal.
WINDOW_CONVENTION: Final = "[start_day, end_day)"


def _g5_window(window: Any, index: int, n_days: int) -> dict[str, Any]:
    """One shaded confounder window, checked so the figure cannot be drawn off the axis."""
    if not isinstance(window, dict):
        raise ArtifactSchemaError(
            "g5_confounder_null", f"windows[{index}] is {type(window).__name__}, not an object"
        )
    missing = [k for k in ("confounder", "start_day", "end_day", "role") if k not in window]
    if missing:
        raise ArtifactSchemaError(
            "g5_confounder_null",
            f"windows[{index}] is missing {missing}. 'role' is required: the dashboard must "
            "read adversarial-vs-control from the artefact, not hardcode which confounders "
            "were the adversarial ones.",
        )
    if window["role"] not in G5_WINDOW_ROLES:
        raise ArtifactSchemaError(
            "g5_confounder_null",
            f"windows[{index}] role {window['role']!r} is not one of {sorted(G5_WINDOW_ROLES)}",
        )
    start, end = int(window["start_day"]), int(window["end_day"])
    if not 0 <= start < end <= n_days:
        raise ArtifactSchemaError(
            "g5_confounder_null",
            f"windows[{index}] spans days [{start}, {end}) — the convention is "
            f"{WINDOW_CONVENTION}, half-open — which is not a non-empty span inside "
            f"[0, {n_days}); a shaded band cannot "
            "fall off the x axis, and a zero-day band is the silent failure T-0112 found "
            "(five of nine confounder windows measured as zero days while every assertion "
            "passed)",
        )
    return {**window, "start_day": start, "end_day": end, "role": str(window["role"])}


def build_g5(series_doc: dict[str, Any]) -> dict[str, Any]:
    """The G5 figure data, from whatever the gate dumped.

    Shape required of the gate output, and checked here so a malformed dump fails at
    generation rather than in a browser: ``prevalence`` (0.0 — every alert in this run is a
    false positive by construction), ``nominal_alert_rate`` (K reviews per day per N
    merchants; a rate that ignores K is decoration), ``n_days``, the confounder ``windows``
    to shade — each half-open, ``[start_day, end_day)``, stated on the payload as
    ``window_convention`` so #62 does not have to guess and shade every band a day too wide
    — and one ``series`` entry per detector — ``raw`` and ``cohort-residual``, the
    two lines whose difference *is* charter hypothesis K-1.

    ``split`` on each series is ``NULL_RUN``: this is a freshly generated zero-prevalence
    population, not the validation or test split, and saying ``VALIDATION`` would be false.
    """
    required = ("prevalence", "nominal_alert_rate", "n_days", "windows", "series")
    missing = [k for k in required if k not in series_doc]
    if missing:
        raise ArtifactSchemaError("g5_confounder_null", f"gate output is missing {missing}")
    n_days = int(series_doc["n_days"])
    if float(series_doc["prevalence"]) != 0.0:
        raise ArtifactSchemaError(
            "g5_confounder_null",
            f"prevalence is {series_doc['prevalence']}, not 0.0. G5 only means anything at "
            "zero prevalence, where every alert is a false positive by construction.",
        )

    windows = [_g5_window(w, i, n_days) for i, w in enumerate(series_doc["windows"])]

    series: list[dict[str, Any]] = []
    for entry in series_doc["series"]:
        rates = list(entry["alert_rate_by_day"])
        if len(rates) != n_days:
            raise ArtifactSchemaError(
                "g5_confounder_null",
                f"detector {entry.get('detector')!r} has {len(rates)} daily rates for "
                f"n_days={n_days}; the x axis and the y series must be the same length",
            )
        series.append(
            {
                "detector": str(entry["detector"]),
                "split": split_label(str(entry.get("split", "null_run"))),
                "threshold": entry.get("threshold"),
                "quiet_day_rate": entry.get("quiet_day_rate"),
                "alert_rate_by_day": rates,
                "window_excess": list(entry.get("window_excess", [])),
                "verdict": entry.get("verdict"),
            }
        )
    return {
        "prevalence": float(series_doc["prevalence"]),
        "nominal_alert_rate": float(series_doc["nominal_alert_rate"]),
        "n_days": n_days,
        "excess_allowed_pp": series_doc.get("excess_allowed_pp"),
        "window_convention": WINDOW_CONVENTION,
        "windows": windows,
        "series": series,
    }


# ─────────────────────────────────────────────────────────────────────────────
# rung_roster — the rungs the ladder cannot show
# ─────────────────────────────────────────────────────────────────────────────


def build_rung_roster(doc: dict[str, Any]) -> dict[str, Any]:
    """Pass the roster YAML through, checked. It is a source file, not a computation.

    ``ladder.json`` is built from ``data/v2/eval/*.json`` and therefore contains exactly the
    rungs that were scored. A rung that was cut, deferred or never started has no row and is
    simply **absent** — not shown as cut, invisible. #64 asks for the opposite, so the
    roster is a hand-maintained file and this function's only job is to refuse a malformed
    one rather than to derive anything.

    Nothing is defaulted. A missing ``status`` is a refusal, not a ``"planned"``: guessing
    on behalf of a document that did not say would produce precisely the confidently-wrong
    roster the file exists to avoid.
    """
    for key in ("statuses", "rungs", "source"):
        if key not in doc:
            raise ArtifactSchemaError("rung_roster", f"roster YAML is missing {key!r}")
    statuses = sorted(str(v) for v in doc["statuses"])
    if set(statuses) != set(RUNG_STATUS_VALUES):
        raise ArtifactSchemaError(
            "rung_roster",
            f"roster YAML declares statuses {statuses}, which is not the contract's "
            f"{sorted(RUNG_STATUS_VALUES)}. The vocabulary is pinned in two places on "
            "purpose: the file the lead edits, and the contract the loader checks.",
        )
    roster = [{str(k): v for k, v in entry.items()} for entry in doc["rungs"]]
    unverified = sorted(str(e["name"]) for e in roster if e.get("status") == "UNVERIFIED")
    return {
        "roster": roster,
        "statuses": statuses,
        "source": doc["source"],
        "n_unverified": len(unverified),
        "unverified": unverified,
        "roster_note": (
            "Derived from committed documents by an agent and NOT yet confirmed by the "
            "lead. Every entry carries the citation it was derived from so the derivation "
            "can be checked rather than trusted. An entry with status UNVERIFIED means the "
            "documents disagree or are silent and must not be rendered as project state. "
            "No entry carries a score: scores live in ladder.json under the same rung id, "
            "and a rung with no ladder row was not scored."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# cost_sweep — the sweep that ran once and was never rendered
# ─────────────────────────────────────────────────────────────────────────────

#: The four pricing arms `scripts/cost_sweep.py` dumps, mapped to the series names the
#: panel plots. `hold_capable` is nested one level deeper (by exposure arm) than the other
#: two, which is why this is a map rather than a loop over the dump's top-level keys.
_SWEEP_ARMS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("declared", ("hold_capable", "declared")),
    ("realised", ("hold_capable", "realised")),
    ("review_only", ("review_only",)),
    ("hold_forbidden", ("hold_forbidden_arm_b",)),
)


def _sweep_at(doc: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    node: Any = doc
    for step in path:
        if not isinstance(node, dict) or step not in node:
            raise ArtifactSchemaError(
                "cost_sweep", f"sweep dump is missing {'.'.join(path)!r} (looked for {step!r})"
            )
        node = node[step]
    if not isinstance(node, dict):
        raise ArtifactSchemaError("cost_sweep", f"{'.'.join(path)!r} is not a ratio -> policy map")
    return node


def build_cost_sweep(doc: dict[str, Any]) -> dict[str, Any]:
    """Reshape the committed sweep dump into one series per policy, per pricing arm.

    The dump is keyed ratio-first (``{"0.01": {"rung4": ...}}``) because that is the order
    it was computed in; a chart needs it policy-first, with every series the same length as
    the shared x-axis. Doing that reshape here rather than in the view is what makes the
    ragged case a **refusal**: a policy absent at one ratio would otherwise render as a
    curve that silently shifts along the axis, which is indistinguishable from a real
    change in the number. Nothing here computes a savings figure — ``scripts/cost_sweep.py``
    owns the measurement and this module never recomputes one.

    ``shipped_ratio_within_grid`` exists because the sweep's own document argues the grid
    must not be extended after seeing the curve. Whether the config's operating point falls
    inside the declared grid is therefore a property a reader is entitled to check on the
    figure, not a claim they have to take from prose.
    """
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        raise ArtifactSchemaError("cost_sweep", "sweep dump is missing its 'meta' block")

    arms: dict[str, list[dict[str, Any]]] = {}
    axis: list[float] | None = None
    for label, path in _SWEEP_ARMS:
        by_ratio = _sweep_at(doc, path)
        ratios = sorted(float(r) for r in by_ratio)
        if axis is None:
            axis = ratios
        elif ratios != axis:
            raise ArtifactSchemaError(
                "cost_sweep",
                f"arm {label!r} was swept over ratio grid {ratios}, but an earlier arm used "
                f"{axis}. A series shorter than the axis plots as a shifted curve, which "
                "reads as a change in the number rather than as missing data.",
            )
        # Keyed by the raw string the dump used, since float(r) round-trips but the key does not.
        raw = {float(r): r for r in by_ratio}
        policies = sorted({p for r in by_ratio for p in by_ratio[r]})
        arms[label] = [
            {"policy": p, "values": [by_ratio[raw[r]].get(p) for r in ratios]} for p in policies
        ]
        for row in arms[label]:
            if any(v is None for v in row["values"]):
                raise ArtifactSchemaError(
                    "cost_sweep",
                    f"arm {label!r} policy {row['policy']!r} has no value at every ratio; a "
                    "gap is a refusal rather than a padded point",
                )

    assert axis is not None  # _SWEEP_ARMS is non-empty, so the loop ran at least once

    # The operating point, resolved ONCE and defensively, because two things downstream
    # depend on it and they must not disagree. `shipped_ratio` is a float the sweep script
    # computes from the fraud-loss distribution, so it can arrive as null (no fraud rows in
    # the window) or NaN (a zero denominator — `cost_sweep.py`'s own markdown renderer
    # guards against exactly that). Either one used to reach `float()` unguarded: null threw
    # an uncaught TypeError past `main`'s ArtifactSchemaError handler, and NaN made every
    # `abs(r - nan)` comparison false so `min` silently returned the FIRST ratio — the
    # decomposition would then be taken at the cheapest point on the grid while the panel
    # printed "taken at the swept ratio nearest the shipped cost matrix". A wrong number
    # under a confident label is the one failure mode this artefact exists to prevent.
    raw_shipped = meta.get("shipped_ratio")
    shipped = (
        float(raw_shipped)
        if isinstance(raw_shipped, (int, float)) and not isinstance(raw_shipped, bool)
        else None
    )
    if shipped is not None and not math.isfinite(shipped):
        shipped = None
    # With no usable operating point there is no "nearest" ratio to be nearest to, so the
    # decomposition is reported at the grid's own midpoint and says so, rather than
    # defaulting to an end of the grid that would read as a choice.
    best = (
        axis.index(min(axis, key=lambda r: abs(r - shipped)))
        if shipped is not None
        else len(axis) // 2
    )

    # What the HOLD action is worth, arm B against the same rows with HOLD unreachable.
    # This is the decomposition the narrative leans on; computing it here keeps it out of
    # the view, where it would become a typed-in constant nobody could re-derive.
    with_hold = {r["policy"]: r["values"] for r in arms["realised"]}
    without_hold = {r["policy"]: r["values"] for r in arms["hold_forbidden"]}
    decomposition = [
        {
            "policy": policy,
            "with_hold": with_hold[policy][best],
            "without_hold": without_hold[policy][best],
            "delta": with_hold[policy][best] - without_hold[policy][best],
        }
        for policy in sorted(set(with_hold) & set(without_hold))
    ]

    return {
        "ratios": axis,
        "arms": arms,
        "meta": meta,
        "shipped_ratio": shipped,
        "shipped_ratio_within_grid": shipped is not None and axis[0] <= shipped <= axis[-1],
        "shipped_ratio_nearest_index": best,
        # The ratio the decomposition was actually taken at, so the panel can print the
        # number instead of the claim "nearest the shipped cost matrix" — which is only
        # true when there was a usable shipped ratio to be nearest to.
        "hold_decomposition_at_ratio": axis[best],
        "hold_decomposition_anchored": shipped is not None,
        "hold_decomposition": decomposition,
        "arm_note": (
            "declared/realised price each policy on the actions it actually takes and may "
            "HOLD; review_only prices every policy the floors' way, REVIEW-only, which is "
            "the only like-for-like comparison against a floor; hold_forbidden is arm B "
            "with HOLD made unreachable and nothing else changed. The decomposition is "
            "taken at the swept ratio nearest the shipped cost matrix."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# journey — the three generations, as committed literals
# ─────────────────────────────────────────────────────────────────────────────


def build_journey(doc: dict[str, Any]) -> dict[str, Any]:
    """Pass the generation waypoints through, checked. Like the roster, a source file.

    Prime Directive 2 closes prior harnesses forever: no G1 or G2 number is recomputed,
    corrected or improved. So these are literals — and the only thing that separates a
    literal from an assertion is the document it was read out of, which is why an entry or
    a figure that cites nothing is a refusal rather than an entry with an empty field.

    ``external`` marks a generation living in a **different repository**. It must name that
    repository: the panel renders the marker as "cited, not recomputed", and a marker with
    nothing behind it is worse than no marker at all. Nothing in this module reads across a
    repository boundary — doing so would break the clean-clone build the charter makes a
    stop-work condition — so an external generation's numbers arrive here the same way the
    rest of this file's inputs do, as committed text.
    """
    for key in ("generations", "source"):
        if key not in doc:
            raise ArtifactSchemaError("journey", f"journey YAML is missing {key!r}")

    generations: list[dict[str, Any]] = []
    for i, raw in enumerate(doc["generations"]):
        if not isinstance(raw, dict):
            raise ArtifactSchemaError("journey", f"generations[{i}] is not an object")
        entry = {str(k): v for k, v in raw.items()}
        for key in ("id", "title", "thesis", "outcome", "citation"):
            if key not in entry:
                raise ArtifactSchemaError("journey", f"generations[{i}] is missing {key!r}")
        if entry["id"] not in GENERATION_IDS:
            raise ArtifactSchemaError(
                "journey",
                f"generations[{i}] has id {entry['id']!r}, not one of {list(GENERATION_IDS)}. "
                "The public labels are pinned so they cannot drift back into the charter's "
                "'v1'/'v2', which name different trees in the two repositories.",
            )
        if not entry["citation"]:
            raise ArtifactSchemaError(
                "journey",
                f"generations[{i}] ({entry['id']}) cites nothing. A prior-generation "
                "number is a literal; a literal without its document is an assertion.",
            )
        external = bool(entry.get("external"))
        if external and not entry.get("source_repo"):
            raise ArtifactSchemaError(
                "journey",
                f"generations[{i}] ({entry['id']}) is marked external but names no "
                "'source_repo'. External means it came from another repository, and which "
                "repository is the whole content of the claim.",
            )
        for j, figure in enumerate(entry.get("figures") or []):
            for key in ("label", "value", "citation"):
                if key not in figure:
                    raise ArtifactSchemaError(
                        "journey", f"generations[{i}].figures[{j}] is missing {key!r}"
                    )
            if not figure["citation"]:
                raise ArtifactSchemaError(
                    "journey",
                    f"generations[{i}].figures[{j}] ({figure['label']!r}) cites nothing.",
                )
        entry["external"] = external
        entry["provenance_note"] = "cited, not recomputed" if external else "built in this tree"
        entry.setdefault("charter_name", None)
        entry.setdefault("figures", [])
        generations.append(entry)

    return {
        "generations": generations,
        "source": doc["source"],
        "n_external": sum(1 for g in generations if g["external"]),
        "naming": {
            "public": list(GENERATION_IDS),
            "charter": {g["id"]: g["charter_name"] for g in generations},
            "note": (
                "The public labels G1/G2/G3 exist because the internal vocabulary collides: "
                "the charter calls the HMM sentinel 'v1' and this tree 'v2', while the "
                "constrained-RL tree in the sibling repository calls itself 'v1'. No charter, "
                "lock or results file was edited to introduce these labels; this map is how "
                "the two vocabularies reconcile."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Emit
# ─────────────────────────────────────────────────────────────────────────────


def _write(out_dir: Path, name: str, doc: dict[str, Any]) -> tuple[Path, str]:
    validate(doc)  # never write a file the loader would reject
    blob = canonical_bytes(doc)
    path = out_dir / f"{name}.json"
    path.write_bytes(blob)
    return path, sha256_bytes(blob)


def build_all(
    root: Path = REPO_ROOT,
    *,
    results_dir: Path | None = None,
    g5_path: Path | None = None,
    roster_path: Path | None = None,
    sweep_path: Path | None = None,
    journey_path: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Emit every artefact this tree has input for. Returns the manifest payload."""
    results = root / (results_dir or DEFAULT_RESULTS_DIR)
    g5_file = root / (g5_path or DEFAULT_G5_PATH)
    roster_file = root / (roster_path or DEFAULT_ROSTER_PATH)
    sweep_file = root / (sweep_path or DEFAULT_SWEEP_PATH)
    journey_file = root / (journey_path or DEFAULT_JOURNEY_PATH)
    target = out_dir if out_dir is not None else root / ARTIFACTS_DIR
    target.mkdir(parents=True, exist_ok=True)

    lock_payload, lock_paths = build_lock_state(root)
    live = next(lock for lock in lock_payload["locks"] if lock["authoritative"])
    base_provenance: dict[str, Any] = {
        "authoritative_lock": lock_payload["authoritative_lock"],
        "eval_lock_sha": live["hashes"]["eval_module_sha256"],
        "open_count": live["open_count"],
        "frozen_at_git_sha": live["frozen_at_git_sha"],
    }

    index: list[dict[str, Any]] = []

    def record(name: str, doc: dict[str, Any] | None, *, split: str | None, reason: str) -> None:
        if doc is None:
            index.append(
                {
                    "name": name,
                    "file": None,
                    "status": "MISSING",
                    "split": split,
                    "sha256": None,
                    "reason": reason,
                }
            )
            return
        path, digest = _write(target, name, doc)
        index.append(
            {
                "name": name,
                "file": path.name,
                "status": "PRESENT",
                "split": split,
                "sha256": digest,
                "reason": None,
            }
        )

    record(
        "lock_state",
        envelope(
            "lock_state",
            lock_payload,
            split=None,
            provenance={**base_provenance, "inputs": _inputs_provenance(lock_paths, root)},
        ),
        split=None,
        reason="",
    )

    if results.is_dir() and any(results.glob("*.json")):
        rows = read_result_rows(results)
        # sanitise() is the backstop: _median already nulls the non-finite values it
        # aggregates, so this catches anything a future field adds without going through it.
        ladder, _ = sanitise(
            build_ladder(rows, test_split_opened=bool(lock_payload["test_split_opened"]))
        )
        ladder["serialisation_note"] = (
            "A null metric means the value was non-finite on every seed. Which token it "
            "was (NaN, Infinity, -Infinity) and on how many seeds is in the same row's "
            "non_finite map — ttd_median_days: null with {'Infinity': 5} beside it reads "
            "'never detected on all five seeds'. JSON.parse accepts neither literal, so "
            "neither is ever emitted."
        )
        splits = ladder["splits"]
        record(
            "ladder",
            envelope(
                "ladder",
                ladder,
                split=splits[0] if len(splits) == 1 else None,
                provenance={
                    **base_provenance,
                    **_results_provenance(rows, lock_payload["locks"]),
                    "inputs": _inputs_provenance(sorted(results.glob("*.json")), root),
                },
            ),
            split=splits[0] if len(splits) == 1 else None,
            reason="",
        )
    else:
        record(
            "ladder",
            None,
            split=None,
            reason=f"no scored results under {_rel(results, root)}; "
            "run `make eval` for at least one rung",
        )

    if g5_file.is_file():
        g5, _ = sanitise(build_g5(_read_json(g5_file)))
        record(
            "g5_confounder_null",
            envelope(
                "g5_confounder_null",
                g5,
                split="NULL_RUN",
                provenance={
                    **base_provenance,
                    "inputs": _inputs_provenance([g5_file], root),
                },
            ),
            split="NULL_RUN",
            reason="",
        )
    else:
        record(
            "g5_confounder_null",
            None,
            split="NULL_RUN",
            reason=f"no G5 gate output at {_rel(g5_file, root)}; "
            "`make gates` must dump the confounder-null series before this figure exists",
        )

    if roster_file.is_file():
        roster = build_rung_roster(yaml.safe_load(roster_file.read_text(encoding="utf-8")))
        record(
            "rung_roster",
            envelope(
                "rung_roster",
                roster,
                split=None,
                provenance={
                    **base_provenance,
                    "inputs": _inputs_provenance([roster_file], root),
                },
            ),
            split=None,
            reason="",
        )
    else:
        record(
            "rung_roster",
            None,
            split=None,
            reason=f"no rung roster at {_rel(roster_file, root)}; a cut rung would then be "
            "absent from the dashboard rather than named as cut",
        )

    if sweep_file.is_file():
        sweep, _ = sanitise(build_cost_sweep(_read_json(sweep_file)))
        record(
            "cost_sweep",
            envelope(
                "cost_sweep",
                sweep,
                split="VALIDATION",
                provenance={**base_provenance, "inputs": _inputs_provenance([sweep_file], root)},
            ),
            split="VALIDATION",
            reason="",
        )
    else:
        record(
            "cost_sweep",
            None,
            split="VALIDATION",
            reason=f"no cost sweep at {_rel(sweep_file, root)}; run "
            "`python scripts/cost_sweep.py` — until it has, every savings number this "
            "project publishes is a single point estimate at one cost matrix",
        )

    if journey_file.is_file():
        journey = build_journey(yaml.safe_load(journey_file.read_text(encoding="utf-8")))
        record(
            "journey",
            envelope(
                "journey",
                journey,
                split=None,
                provenance={**base_provenance, "inputs": _inputs_provenance([journey_file], root)},
            ),
            split=None,
            reason="",
        )
    else:
        record(
            "journey",
            None,
            split=None,
            reason=f"no generation waypoints at {_rel(journey_file, root)}; the panel would "
            "then present this tree with no account of the two generations before it",
        )

    manifest_payload = {
        "contract": "rakshak-v3-dashboard",
        "artifacts": sorted(index, key=lambda a: str(a["name"])),
    }
    _write(
        target,
        "manifest",
        envelope("manifest", manifest_payload, split=None, provenance=base_provenance),
    )
    return manifest_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rakshak-artifacts", description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--g5", type=Path, default=None)
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--sweep", type=Path, default=None)
    parser.add_argument("--journey", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        manifest = build_all(
            args.root,
            results_dir=args.results_dir,
            g5_path=args.g5,
            roster_path=args.roster,
            sweep_path=args.sweep,
            journey_path=args.journey,
            out_dir=args.out,
        )
    except ArtifactSchemaError as exc:
        print(f"REFUSED  {exc}", file=sys.stderr)
        return 1

    print(f"artefact contract {SCHEMA_VERSION}")
    for entry in manifest["artifacts"]:
        mark = "ok     " if entry["status"] == "PRESENT" else "MISSING"
        detail = entry["file"] if entry["status"] == "PRESENT" else entry["reason"]
        print(f"  {mark} {entry['name']:<20} {detail}")
    return 0


if __name__ == "__main__":  # `make artifacts` -> python -m rakshak.artifacts.build
    raise SystemExit(main())
