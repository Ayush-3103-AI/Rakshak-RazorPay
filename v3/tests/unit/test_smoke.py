"""T-100: the scaffold is importable and the environment is the pinned one."""

from __future__ import annotations

import sys

import rakshak


def test_version_is_v2() -> None:
    assert rakshak.__version__ == "2.0.0"


def test_schema_version_is_an_int() -> None:
    assert isinstance(rakshak.SCHEMA_VERSION, int)


def test_python_is_311() -> None:
    # polars, duckdb and lightgbm are all pinned against 3.11. A 3.12 env resolves to
    # different wheels and `make all` stops being a reproducibility claim.
    assert sys.version_info[:2] == (3, 11)


def test_pandas_is_not_installed() -> None:
    # CLAUDE.md rejects pandas outright: polars offline, numpy online. Catching it here
    # means a stray `import pandas` fails at the dependency, not in review.
    with __import__("pytest").raises(ImportError):
        __import__("pandas")
