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
result rows' own ``eval_lock_sha`` and ``git_sha`` beside the authoritative lock's, and a
row computed against a different eval module is a refusal. What the rows cannot certify —
the generator and scenario-config hashes, which ``EvalResult`` does not record — is said
so in ``harness_note`` rather than left for the reader to assume.

**Lock files are discovered, never enumerated.** ``EVAL-LOCK.json`` is cycle 1 and
``EVAL-LOCK-CYCLE2.json`` supersedes it; a further cycle is expected and must need no code
change here, so nothing names a lock file. The artefact globs ``EVAL-LOCK*.json``, orders
by cycle, resolves the supersession chain from each file's own ``supersedes``, and makes
``authoritative_lock`` an explicit field so the dashboard never has to infer which lock is
live. Exactly one lock may be unsuperseded; two would mean the chain is broken, and that
is a refusal.
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

from rakshak.artifacts import (
    ARTIFACTS_DIR,
    SCHEMA_VERSION,
    ArtifactSchemaError,
    canonical_bytes,
    envelope,
    sanitise,
    sha256_bytes,
    split_label,
    validate,
)
from rakshak.schemas import EvalResult

__all__ = ["build_all", "build_g5", "build_ladder", "build_lock_state", "main"]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]

DEFAULT_RESULTS_DIR: Final = Path("data/v2/eval")
#: Where the G5 gate is expected to drop its series. Nothing in this module writes it —
#: `tests/gates/test_g5_confounder_null.py` owns the measurement and must own the dump.
DEFAULT_G5_PATH: Final = Path("data/v2/gates/g5_series.json")

LOCK_GLOB: Final = "EVAL-LOCK*.json"
LOCK_HASH_KEYS: Final = (
    "eval_module_sha256",
    "generator_module_sha256",
    "scenario_config_sha256",
)

#: Every float field on ``EvalResult``, derived from the dataclass so that a metric added
#: upstream reaches the dashboard without this file being edited.
_NON_METRIC: Final = frozenset(
    {"rung", "split", "recall_by_typology", "floor_fail", "eval_lock_sha", "git_sha", "open_count"}
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


def _results_provenance(rows: Sequence[dict[str, Any]], live: dict[str, Any]) -> dict[str, Any]:
    """What the *rows* say about themselves, beside what the live lock says about itself.

    Without this the ladder carries the authoritative lock's shas and nothing else, which
    reads as "these numbers were computed under this lock" — and that is exactly the claim
    the artefact is not entitled to make. A result row records the harness hash and the
    commit it ran at; it does not record the generator or scenario-config hash, so it
    cannot certify the *geometry* the population was drawn at. The dashboard gets both
    facts and the gap between them, stated rather than papered over.

    The harness hash it *can* check is checked, and a mismatch is a refusal: a number
    computed against a different eval module is not comparable to one computed against
    this one, which is the whole reason ``EVAL-LOCK`` exists.
    """
    result_lock_shas = sorted({str(r["eval_lock_sha"]) for r in rows})
    authoritative = str(live["hashes"]["eval_module_sha256"])
    strays = [sha for sha in result_lock_shas if sha != authoritative]
    if strays:
        raise ArtifactSchemaError(
            "ladder",
            f"result rows were computed against eval module {strays} but the authoritative "
            f"lock {live['file']} freezes {authoritative}. Rendering them under this lock "
            "would attribute a number to a harness that did not produce it; re-run `make "
            "eval` against the live lock.",
        )
    return {
        "results_eval_lock_sha": result_lock_shas,
        "results_git_sha": sorted({str(r["git_sha"]) for r in rows}),
        "results_open_count": sorted({int(r["open_count"]) for r in rows}),
        "harness_note": (
            "results_eval_lock_sha is verified equal to the authoritative lock's "
            "eval_module_sha256 — the harness that produced these numbers is the frozen "
            "one. EvalResult rows do NOT record generator_module_sha256 or "
            "scenario_config_sha256, so this artefact cannot certify that they were "
            "computed at the authoritative lock's geometry. Compare results_git_sha with "
            "the lock's frozen_at_git_sha before presenting a number as current."
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

    live = [e for e in entries if e["superseded_by"] is None]
    if len(live) != 1:
        raise ArtifactSchemaError(
            "lock_state",
            f"expected exactly one unsuperseded lock, found {[e['file'] for e in live]}. "
            "The dashboard must not have to infer which lock is authoritative.",
        )
    live[0]["authoritative"] = True
    if live[0]["cycle"] != max(e["cycle"] for e in entries):
        raise ArtifactSchemaError(
            "lock_state",
            f"{live[0]['file']} is unsuperseded but is not the highest cycle; the "
            "supersession chain and the cycle numbers disagree",
        )

    payload = {
        "authoritative_lock": live[0]["file"],
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
    if not 0 <= start <= end < n_days:
        raise ArtifactSchemaError(
            "g5_confounder_null",
            f"windows[{index}] spans days [{start}, {end}], which is not inside "
            f"[0, {n_days - 1}]; a shaded band cannot fall off the x axis",
        )
    return {**window, "start_day": start, "end_day": end, "role": str(window["role"])}


def build_g5(series_doc: dict[str, Any]) -> dict[str, Any]:
    """The G5 figure data, from whatever the gate dumped.

    Shape required of the gate output, and checked here so a malformed dump fails at
    generation rather than in a browser: ``prevalence`` (0.0 — every alert in this run is a
    false positive by construction), ``nominal_alert_rate`` (K reviews per day per N
    merchants; a rate that ignores K is decoration), ``n_days``, the confounder ``windows``
    to shade, and one ``series`` entry per detector — ``raw`` and ``cohort-residual``, the
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
        "windows": windows,
        "series": series,
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
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Emit every artefact this tree has input for. Returns the manifest payload."""
    results = root / (results_dir or DEFAULT_RESULTS_DIR)
    g5_file = root / (g5_path or DEFAULT_G5_PATH)
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
                    **_results_provenance(rows, live),
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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        manifest = build_all(
            args.root, results_dir=args.results_dir, g5_path=args.g5, out_dir=args.out
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
