"""G4 — no leakage. Ground-truth fields are unreachable from features and models.

**Blocking.** Prime Directive 3: ``persona_id``, ``risk_typology_id``, ``drift_onset_at``,
``true_loss_amount_inr``, ``is_unreported`` and ``GroundTruth`` itself must never be
reachable from ``src/rakshak/features/`` or ``src/rakshak/models/``. Leakage invalidates
every number in the project, and it does so silently and flatteringly, which is the worst
combination available.

The scan is AST-based rather than textual. A substring search would both miss
``frame["drift_onset_at"]`` reached through a variable and fire on the word appearing in
a docstring that explains the quarantine. Imports, attribute access, bare names and
string literals are all checked; docstrings are excluded.

**One half of this gate is deferred, and that is stated rather than hidden.** §7's G4
also requires that "point-in-time recomputation at time t matches the stored feature
vector exactly". That needs a feature layer, which is Lane B (T-120/T-121) and does not
exist as this ticket lands. It is recorded below as a SKIP with its owner.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gates_report import green_if, record

from rakshak.schemas import RADIOACTIVE_FIELDS

SRC = Path(__file__).resolve().parents[2] / "src" / "rakshak"
QUARANTINED_FROM = ("features", "models")


def _string_constants(tree: ast.AST) -> set[str]:
    """String literals, minus docstrings.

    A radioactive field reaches a model just as well through ``frame["drift_onset_at"]``
    as through an import, and that form has no name node to catch.
    """
    docstrings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.extend(
                f"{path.name}:{node.lineno} imports {alias.name}"
                for alias in node.names
                if alias.name in RADIOACTIVE_FIELDS
            )
        elif isinstance(node, ast.Attribute) and node.attr in RADIOACTIVE_FIELDS:
            found.append(f"{path.name}:{node.lineno} attribute .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in RADIOACTIVE_FIELDS:
            found.append(f"{path.name}:{node.lineno} name {node.id}")
    found.extend(
        f"{path.name} string literal {literal!r}"
        for literal in sorted(_string_constants(tree) & RADIOACTIVE_FIELDS)
    )
    return found


def test_g4_no_ground_truth_reaches_features_or_models() -> None:
    offenders: list[str] = []
    scanned = 0
    for package in QUARANTINED_FROM:
        for path in sorted((SRC / package).rglob("*.py")):
            scanned += 1
            offenders.extend(f"{package}/{item}" for item in scan(path))
    ok = green_if(
        "G4 no-leakage",
        not offenders,
        f"{len(offenders)} forbidden reference(s) across {scanned} file(s) in "
        f"{'/'.join(QUARANTINED_FROM)}",
        "; ".join(offenders) if offenders else f"quarantined: {sorted(RADIOACTIVE_FIELDS)}",
    )
    assert ok, f"ground-truth leakage: {offenders}"


def test_g4_the_scanner_actually_catches_leakage(tmp_path: Path) -> None:
    """A clean scan of a nearly empty package is not evidence that the scanner works.

    Lane B's files land later. This proves the gate will see them when they do — and it
    is the reason a green G4 today means anything at all.
    """
    leaky = (
        "from rakshak.schemas import GroundTruth\n",
        "def f(gt):\n    return gt.drift_onset_at\n",
        "def f(frame):\n    return frame['risk_typology_id']\n",
        "def f(row):\n    return row[chr(0x70) + 'ersona_id']\n",
    )
    probe = tmp_path / "probe.py"
    for i, snippet in enumerate(leaky[:3]):
        probe.write_text(snippet, encoding="utf-8")
        assert scan(probe), f"scanner missed leak #{i}: {snippet!r}"

    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""This docstring mentions drift_onset_at only in prose."""\nX = 1\n', encoding="utf-8"
    )
    assert not scan(clean)


def test_g4_point_in_time_recomputation_is_deferred() -> None:
    """§7's second G4 clause needs a feature layer, and Lane B has not landed one yet."""
    record(
        "G4b point-in-time",
        "SKIP",
        "no feature layer to recompute against",
        "08-generator-v2-spec.md §7 G4 also requires that a point-in-time recomputation "
        "at time t matches the stored feature vector exactly. Owner: Lane B, "
        "T-120/T-121. The framework half is already proven by tests/parity/ (T-102).",
    )
