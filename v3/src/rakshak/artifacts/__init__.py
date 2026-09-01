"""The v3 artefact contract — the dashboard's only data source (T-0126, #60).

Committed, versioned, schema-checked JSON under ``artifacts/``. **No backend, ever.**

This module is the contract. ``build.py`` is the generator that emits against it. The two
are separate so that the validator can be run over a file nobody in this process wrote —
which is the only way to be sure the loader's check and the emitter's check are the same
check.

Four properties, and none of them is a matter of discipline:

**Every artefact is versioned.** ``schema_version`` is on the envelope of every file. The
loader (#61) rejects a mismatch by name and reason. Bump ``SCHEMA_VERSION`` whenever the
shape of any payload changes; the dashboard pins the exact string.

**Every number carries its split, as a field.** Not as a filename convention, not in a
footnote. ``VALIDATION`` until T-0116 opens the test split, and the site must never render
an unlocked number in the visual register of a final one. ``NULL_RUN`` is the honest
fourth value: the G5 confounder null is measured on a freshly generated zero-prevalence
population and belongs to no split at all. Calling it ``VALIDATION`` would be a lie of
convenience, and inventing the value costs one line.

**Serialisation is canonical.** ``sort_keys=True`` kills dict-ordering drift, no wall-clock
timestamp is ever stamped into an output, and ``allow_nan=False`` makes a bare ``NaN`` a
hard error rather than a token no ``JSON.parse`` on earth accepts. Non-finite floats are
therefore mapped to ``null`` and the *reason* is recorded alongside in a ``non_finite``
map — ``ttd_median_days: null`` with ``{"Infinity": 5}`` beside it says "never detected on
all five seeds", which is a finding; a bare ``null`` says nothing.

**Nothing radioactive leaves.** ``validate`` rejects any payload containing a key in
``FORBIDDEN_KEYS`` at any depth. ``recall_by_typology`` survives that scan deliberately:
per-typology recall is an *aggregate metric the harness is required to report* (#43), not
a per-merchant ground-truth field, and the keys are typology names rather than a
``risk_typology_id`` column.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from rakshak.schemas import RADIOACTIVE_FIELDS

__all__ = [
    "ARTIFACT_NAMES",
    "ARTIFACTS_DIR",
    "FORBIDDEN_KEYS",
    "RUNG_STATUS_VALUES",
    "SCHEMA_VERSION",
    "SPLIT_VALUES",
    "ArtifactSchemaError",
    "ArtifactSplit",
    "canonical_bytes",
    "envelope",
    "sanitise",
    "sha256_bytes",
    "split_label",
    "validate",
    "validate_file",
]

#: Bump on any payload shape change. The loader pins this exact string and rejects
#: anything else by artefact name and reason.
SCHEMA_VERSION: Final = "v3.1.0"

#: Committed. This is the point: the panel can open the same bytes the site serves.
ARTIFACTS_DIR: Final = Path("artifacts")

ArtifactSplit = Literal["TRAIN", "VALIDATION", "TEST", "NULL_RUN"]

SPLIT_VALUES: Final[frozenset[str]] = frozenset({"TRAIN", "VALIDATION", "TEST", "NULL_RUN"})

#: ``EvalResult.split`` is lowercase and terse; the contract is loud on purpose.
_SPLIT_MAP: Final[dict[str, str]] = {
    "train": "TRAIN",
    "val": "VALIDATION",
    "validation": "VALIDATION",
    "test": "TEST",
    "null_run": "NULL_RUN",
}

#: Prime Directive 3, at the artefact boundary. ``RADIOACTIVE_FIELDS`` already names the
#: generator's ground-truth columns; the extra two are the spellings #60 calls out that
#: the generator does not itself use.
FORBIDDEN_KEYS: Final[frozenset[str]] = RADIOACTIVE_FIELDS | {
    "true_loss_amount",
    "ground_truth",
}

#: name -> the payload keys the loader is entitled to assume exist.
ARTIFACT_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "manifest": ("contract", "artifacts"),
    "lock_state": ("authoritative_lock", "locks"),
    "ladder": ("rungs", "capacity_k", "metric_keys"),
    "g5_confounder_null": (
        "prevalence",
        "nominal_alert_rate",
        "n_days",
        "window_convention",
        "windows",
        "series",
    ),
    "rung_roster": ("roster", "statuses", "source"),
}

#: The roster's vocabulary, pinned here rather than in the YAML alone so a hand-edited
#: status is a named refusal instead of an unrecognised badge on a judge-facing page.
#: ``UNVERIFIED`` is uppercase because it is the one a reader must not skim past: it means
#: the committed documents disagree or are silent, and the entry is a question, not state.
RUNG_STATUS_VALUES: Final[frozenset[str]] = frozenset(
    {"planned", "built", "scored", "cut", "deferred", "conditional", "UNVERIFIED"}
)

#: Artefacts whose rows are numbers, and therefore must carry a split on every row.
_ROW_KEY: Final[dict[str, str]] = {"ladder": "rungs", "g5_confounder_null": "series"}


class ArtifactSchemaError(ValueError):
    """An artefact does not satisfy the contract. Always names the artefact and the reason.

    Raised by the *emitter* as well as the validator, so a file the loader would reject is
    never written in the first place — which is the whole of this ticket's job.
    """

    def __init__(self, artifact: str, reason: str) -> None:
        super().__init__(f"artifact {artifact!r}: {reason}")
        self.artifact = artifact
        self.reason = reason


def split_label(raw: str) -> ArtifactSplit:
    """``"val"`` -> ``"VALIDATION"``. Unknown split is a refusal, never a guess."""
    label = _SPLIT_MAP.get(raw.strip().lower())
    if label is None:
        raise ArtifactSchemaError(
            "<split>", f"unknown split {raw!r}; expected one of {sorted(SPLIT_VALUES)}"
        )
    return label  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────


def sanitise(value: Any) -> tuple[Any, dict[str, dict[str, int]]]:
    """Replace every non-finite float with ``null``, returning what was replaced and why.

    ``JSON.parse`` accepts neither ``NaN`` nor ``Infinity``; Python's ``json`` emits both
    by default. The generator's inputs are full of them — ``ttd_median_days: Infinity``
    means "never detected", ``precision_at_k: NaN`` means "no alerts to be precise about"
    — and both are findings. Dropping the distinction into an untyped ``null`` would throw
    away the most interesting cell in the table, so the reason travels beside the value.

    The returned map is keyed by the *leaf key name*, which is enough for the ladder (one
    metric per name per row) and is where the caller attaches it.
    """
    found: dict[str, dict[str, int]] = {}

    def walk(node: Any, key: str) -> Any:
        if isinstance(node, dict):
            return {str(k): walk(v, str(k)) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [walk(v, key) for v in node]
        if isinstance(node, float) and not math.isfinite(node):
            token = "NaN" if math.isnan(node) else ("Infinity" if node > 0 else "-Infinity")
            found.setdefault(key, {})[token] = found.setdefault(key, {}).get(token, 0) + 1
            return None
        return node

    return walk(value, ""), found


def canonical_bytes(payload: Any) -> bytes:
    """The one serialisation. Sorted keys, two-space indent, LF, trailing newline.

    ``allow_nan=False`` on purpose: reaching here with a non-finite float means ``sanitise``
    was skipped, and a loud ``ValueError`` at emit time beats a file that only fails in the
    browser. Nothing here reads the clock — regenerating from the same inputs is
    byte-identical, and that is an acceptance criterion rather than a nicety.
    """
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def envelope(
    name: str, payload: dict[str, Any], *, split: str | None, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Wrap a payload in the versioned envelope every artefact file carries."""
    return {
        "artifact": name,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "provenance": provenance,
        "payload": payload,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation — run by the emitter before writing, and by the test over what was written
# ─────────────────────────────────────────────────────────────────────────────


def _forbidden_hit(node: Any) -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) in FORBIDDEN_KEYS:
                return str(key)
            hit = _forbidden_hit(value)
            if hit:
                return hit
    elif isinstance(node, list):
        for item in node:
            hit = _forbidden_hit(item)
            if hit:
                return hit
    return None


def _validate_roster(payload: dict[str, Any]) -> None:
    """The roster's own rules. Every entry states a status and cites where it came from.

    Two refusals, and both exist because this file is rendered to judges as project state:

    **No entry may carry a score.** Rungs 5-8 were never scored, and the failure mode is an
    artefact that claims otherwise — a ``metrics`` block full of nulls renders as zeroes on
    a chart and is indistinguishable from "measured, and it was zero". Scores live in
    ``ladder.json``, keyed by the same ``rung`` id; the roster says what exists, the ladder
    says what it measured, and nothing has to reconcile two copies of a number.

    **Every entry cites a document.** A status with no citation is an assertion, and the
    roster was derived by reading committed files rather than by knowing the answer.
    """
    for key in ("roster", "statuses"):
        if not isinstance(payload.get(key), list):
            raise ArtifactSchemaError("rung_roster", f"payload[{key!r}] must be a list")
    declared = set(payload["statuses"])
    if declared != set(RUNG_STATUS_VALUES):
        raise ArtifactSchemaError(
            "rung_roster",
            f"declared statuses {sorted(declared)} do not match the contract's "
            f"{sorted(RUNG_STATUS_VALUES)}",
        )
    for i, entry in enumerate(payload["roster"]):
        if not isinstance(entry, dict):
            raise ArtifactSchemaError("rung_roster", f"roster[{i}] is not an object")
        for key in ("rung", "name", "status", "citation"):
            if key not in entry:
                raise ArtifactSchemaError("rung_roster", f"roster[{i}] is missing {key!r}")
        if entry["status"] not in RUNG_STATUS_VALUES:
            raise ArtifactSchemaError(
                "rung_roster",
                f"roster[{i}] ({entry['name']!r}) has status {entry['status']!r}, not one of "
                f"{sorted(RUNG_STATUS_VALUES)}",
            )
        if not entry["citation"]:
            raise ArtifactSchemaError(
                "rung_roster",
                f"roster[{i}] ({entry['name']!r}) cites nothing. Every entry names the "
                "document it was derived from, or it is an assertion rather than a reading.",
            )
        if "metrics" in entry:
            raise ArtifactSchemaError(
                "rung_roster",
                f"roster[{i}] ({entry['name']!r}) carries a 'metrics' key. The roster says "
                "which rungs exist and what happened to them; it never carries a score. A "
                "rung that was not scored must be ABSENT from ladder.json, not present here "
                "with nulls that render as zeroes.",
            )


def validate(doc: Any) -> str:
    """Check one whole artefact document. Returns its name; raises with name and reason.

    Deliberately duplicates what the JS loader will check, in Python, so that CI can assert
    the emitter and the loader agree on a file neither of them produced.
    """
    if not isinstance(doc, dict):
        raise ArtifactSchemaError("<unknown>", f"top level is {type(doc).__name__}, not an object")
    name = doc.get("artifact")
    if not isinstance(name, str) or name not in ARTIFACT_NAMES:
        raise ArtifactSchemaError(
            str(name), f"unknown artifact name; expected one of {sorted(ARTIFACT_NAMES)}"
        )
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ArtifactSchemaError(
            name, f"schema_version {version!r} does not match contract {SCHEMA_VERSION!r}"
        )
    for key in ("split", "provenance", "payload"):
        if key not in doc:
            raise ArtifactSchemaError(name, f"envelope is missing {key!r}")
    if doc["split"] is not None and doc["split"] not in SPLIT_VALUES:
        raise ArtifactSchemaError(
            name, f"split {doc['split']!r} is not one of {sorted(SPLIT_VALUES)} or null"
        )
    if not isinstance(doc["provenance"], dict) or not isinstance(doc["payload"], dict):
        raise ArtifactSchemaError(name, "provenance and payload must both be objects")

    payload: dict[str, Any] = doc["payload"]
    for key in ARTIFACT_NAMES[name]:
        if key not in payload:
            raise ArtifactSchemaError(name, f"payload is missing required key {key!r}")

    row_key = _ROW_KEY.get(name)
    if row_key is not None:
        rows = payload[row_key]
        if not isinstance(rows, list):
            raise ArtifactSchemaError(name, f"payload[{row_key!r}] must be a list of rows")
        for i, row in enumerate(rows):
            got = row.get("split") if isinstance(row, dict) else None
            if got not in SPLIT_VALUES:
                raise ArtifactSchemaError(
                    name,
                    f"{row_key}[{i}] has split {got!r}; every numeric row carries its split as "
                    "a field, not as a filename convention",
                )

    if name == "rung_roster":
        _validate_roster(payload)

    hit = _forbidden_hit(payload)
    if hit is not None:
        raise ArtifactSchemaError(
            name, f"contains ground-truth field {hit!r} (Prime Directive 3); artefacts are public"
        )
    # Reached here means it also serialises: same call the writer makes.
    canonical_bytes(doc)
    return name


def validate_file(path: Path) -> str:
    """Validate an artefact on disk. Any failure names the file and the reason."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # a truncated write must not read as an empty chart
        raise ArtifactSchemaError(path.name, f"is not parseable JSON: {exc}") from exc
    return validate(doc)
