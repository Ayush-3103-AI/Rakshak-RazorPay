"""Pytest wiring for the parity suite.

The harness itself is in ``parity_harness.py`` so that ``tests/gates/`` can import it
too; this module re-exports it so existing ``from conftest import ...`` lines keep
working, and adds the fixture, which is pytest-only.
"""

from __future__ import annotations

import pytest
from parity_harness import (  # noqa: F401  (re-exported for the parity suite)
    ParityFailure,
    assert_parity,
    end_of_day,
    epochs_between,
    synthetic_stream,
    to_frame,
)

from rakshak.schemas import MerchantProfile, Transaction


@pytest.fixture
def stream(rng: object) -> tuple[list[Transaction], dict[str, MerchantProfile]]:
    return synthetic_stream(rng)
