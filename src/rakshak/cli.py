"""Shared argument-parser helper.

Every script in this repo takes --seed (CLAUDE.md: "Determinism is a hard
requirement"). Building the parser here is how that convention is enforced
rather than remembered.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from rakshak.config import SEED


def base_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser that already carries --seed.

    Args:
        description: Shown in --help.

    Returns:
        Parser with a `--seed` int argument defaulting to `config.SEED`.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED}).",
    )
    return parser


def seed_everything(seed: int) -> np.random.Generator:
    """Seed the stdlib and legacy numpy RNGs, and return a fresh Generator.

    Prefer the returned Generator in new code; the global seeding exists only
    for third-party libraries that reach for the legacy global state.

    Args:
        seed: The seed to apply.

    Returns:
        A `numpy.random.Generator` seeded with `seed`.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
