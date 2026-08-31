"""Pytest wiring for the parity suite. The harness itself is `parity_harness.py`.

This module deliberately re-exports NOTHING. It used to, so that Lane B's existing
`from conftest import ...` lines kept working after the harness moved — and that shim
was itself a bug: `conftest` is importable by name from every directory on the path, so
with both `tests/parity` and `tests/gates` collected in one invocation, the parity
tests' `from conftest import ...` resolved to the GATES conftest and collection failed.
It stayed hidden because `make` runs the two as separate targets.

Nothing anywhere imports `conftest` by name now. Import `parity_harness` instead.
"""

from __future__ import annotations

import pytest
from parity_harness import synthetic_stream

from rakshak.schemas import MerchantProfile, Transaction


@pytest.fixture
def stream(rng: object) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    return synthetic_stream(rng)
