"""AST gate: ``eval.splits.available_labels`` is the ONLY path to the label table.

T-130's done-when clause. The label table is the one input whose careless read destroys
every number downstream — ``label_available_at <= as_of`` is a filter that is written
correctly in eleven call sites and forgotten in the twelfth, so there is exactly one call
site and this test asserts it stays that way.

**What this scan catches** (statically, over every module under ``src/rakshak/``):

- a literal parquet path whose filename mentions labels, anywhere but ``splits.py``;
- a ``scan_parquet`` / ``read_parquet`` call whose argument names a label source;
- SQL text selecting ``FROM ... label ...`` (the duckdb back door);
- any mention of the gate column ``label_available_at`` outside ``splits.py`` (the door),
  ``schemas.py`` (which defines the name) and ``generator/`` (which writes the value) —
  a fourth module handling that column *is* a second gate;
- ``features/**`` touching labels at all, in any form.

**What it cannot catch**, and is therefore left to review:

- a path assembled at runtime (``Path(cfg["data"]) / (kind + ".parquet")``);
- a frame legitimately obtained from ``available_labels`` and then re-exported by a
  helper that re-widens it (e.g. passing ``include_censored=True`` and forgetting to
  count), which is a semantic leak, not a syntactic one;
- anything reached through ``getattr``, ``eval``, ``importlib``, or a duckdb SQL string
  built by concatenation;
- test and notebook code outside ``src/``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "rakshak"

#: The single permitted door, relative to SRC.
DOOR = Path("eval/splits.py")

#: Modules allowed to name the gate column, and why each one is not a second door:
#:
#: - ``eval/splits.py`` — IS the door.
#: - ``schemas.py`` — *defines* the label shape (dataclass field, polars schema key).
#:   Defining a column name is not reading the table, and the point of schemas.py is that
#:   the name exists in exactly one place.
#: - ``generator/`` — **writes** the column. The producer must set the value it emits.
#:   The generator is separately banned from *reading* the label table by
#:   ``test_feature_layer_never_reads_labels``, which is the half that matters: a writer
#:   that reads its own output back is how a delayed label becomes an instant one.
GATE_COLUMN_EXEMPT = {DOOR, Path("schemas.py")}
GATE_COLUMN_EXEMPT_PACKAGES = {"generator"}

_LABEL_PARQUET = re.compile(r"label[a-z_]*\.parquet", re.IGNORECASE)
_LABEL_SQL = re.compile(r"\bfrom\s+[\"'`]?[\w.]*label", re.IGNORECASE)
_READERS = {"scan_parquet", "read_parquet", "scan_pyarrow_dataset", "read_database"}


def _modules() -> list[tuple[Path, ast.Module, str]]:
    out = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        out.append((path.relative_to(SRC), ast.parse(text, filename=str(path)), text))
    return out


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """ids of Constant nodes that are docstrings — prose, not code."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _string_constants(tree: ast.Module) -> list[str]:
    skip = _docstring_nodes(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
    ]


# ─────────── negative controls: prove the detectors actually fire ───────────
#
# A scan is only worth its runtime if it can fail. Each detector is run against a
# synthetic offending module inline, so the controls need no file mutation and no other
# lane's file.

_OFFENDERS = {
    "literal path": ('P = "data/v2/labels.parquet"', _LABEL_PARQUET),
    "sql": ('Q = "SELECT * FROM labels WHERE 1"', _LABEL_SQL),
}


@pytest.mark.parametrize(("name", "case"), list(_OFFENDERS.items()))
def test_regex_detectors_fire_on_a_synthetic_offender(
    name: str, case: tuple[str, re.Pattern[str]]
) -> None:
    source, pattern = case
    tree = ast.parse(source)
    assert any(pattern.search(s) for s in _string_constants(tree)), name


def test_reader_detector_fires_on_a_synthetic_offender() -> None:
    tree = ast.parse('import polars as pl\nf = pl.scan_parquet(LABEL_PATH)\n')
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
        in _READERS
        and "label" in " ".join(ast.unparse(a) for a in n.args).lower()
    ]
    assert hits, "the reader detector would not have caught pl.scan_parquet(LABEL_PATH)"


def test_docstrings_are_not_treated_as_code() -> None:
    """store.py points at the gate in prose. Prose is not a second door."""
    tree = ast.parse('"""we filter label_available_at <= as_of elsewhere."""\nx = 1\n')
    assert "label_available_at" not in " ".join(_string_constants(tree))


def test_modules_were_found() -> None:
    """A scan that silently finds nothing is a green test that guards nothing."""
    mods = _modules()
    assert len(mods) >= 5, f"expected the src tree, found {len(mods)} modules"
    assert any(rel == DOOR for rel, _, _ in mods), f"{DOOR} not found — did the door move?"


def test_no_literal_label_parquet_outside_the_door() -> None:
    offenders = [
        (rel, s)
        for rel, tree, _ in _modules()
        if rel != DOOR
        for s in _string_constants(tree)
        if _LABEL_PARQUET.search(s)
    ]
    assert not offenders, (
        "a literal label parquet path outside eval/splits.py is a second door into the "
        f"label table; read it through available_labels(as_of) instead: {offenders}"
    )


def test_no_label_sql_outside_the_door() -> None:
    offenders = [
        (rel, s)
        for rel, tree, _ in _modules()
        if rel != DOOR
        for s in _string_constants(tree)
        if _LABEL_SQL.search(s)
    ]
    assert not offenders, f"SQL selecting from a label table outside the door: {offenders}"


def test_no_parquet_reader_named_label_outside_the_door() -> None:
    offenders: list[tuple[Path, str]] = []
    for rel, tree, _ in _modules():
        if rel == DOOR:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in _READERS:
                continue
            arg_text = " ".join(ast.unparse(a) for a in node.args)
            if "label" in arg_text.lower():
                offenders.append((rel, ast.unparse(node)))
    assert not offenders, f"a parquet reader pointed at labels outside the door: {offenders}"


def test_gate_column_appears_only_behind_the_door() -> None:
    """``label_available_at`` is the gate itself. Two modules touching it means two gates."""
    offenders = []
    for rel, tree, _ in _modules():
        if rel in GATE_COLUMN_EXEMPT or rel.parts[0] in GATE_COLUMN_EXEMPT_PACKAGES:
            continue
        # Docstrings and comments are prose: store.py's docstring names the gate in order
        # to point at it, which is the opposite of opening a second one.
        if "label_available_at" in _string_constants(tree):
            offenders.append(rel)
        if any(
            isinstance(n, ast.Attribute) and n.attr == "label_available_at" for n in ast.walk(tree)
        ):
            offenders.append(rel)
    assert not offenders, (
        "label_available_at is the point-in-time gate; a module other than eval/splits.py "
        f"(and schemas.py, which only names it) handling it is a second gate: {offenders}"
    )


@pytest.mark.parametrize("package", ["features", "generator"])
def test_feature_layer_never_reads_labels(package: str) -> None:
    """The feature layer has no business knowing an outcome. The generator *writes*
    labels but must not read them back through anything but its own emission path — so
    only reader calls are banned here, not the word ``label``."""
    offenders: list[tuple[Path, str]] = []
    for rel, tree, _ in _modules():
        if rel.parts[0] != package:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rakshak.eval.splits":
                offenders.append((rel, ast.unparse(node)))
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                args = " ".join(ast.unparse(a) for a in node.args).lower()
                if name in _READERS and "label" in args:
                    offenders.append((rel, ast.unparse(node)))
    assert not offenders, f"{package}/ reaches the label table: {offenders}"
