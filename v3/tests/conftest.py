"""Shared fixtures. Seeds are explicit and threaded everywhere — there is no module-level
RNG in src/ and there is none here either."""

from __future__ import annotations

import numpy as np
import pytest

SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """The only RNG a test should use. Every stochastic function in src/ takes one of
    these as an argument, so a test that needs randomness passes this in rather than
    reaching for np.random.*."""
    return np.random.default_rng(SEED)
