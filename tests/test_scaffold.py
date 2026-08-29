"""T-0001 smoke tests: the scaffold is importable, seeded, and deterministic."""

from __future__ import annotations

from rakshak.cli import base_parser, seed_everything
from rakshak.config import COST_REVIEW_INR, RESULTS_DIR, ROOT_DIR, SEED


def test_seed_is_importable() -> None:
    assert SEED == 42


def test_paths_resolve_under_repo_root() -> None:
    assert (ROOT_DIR / "pyproject.toml").exists()
    assert RESULTS_DIR.parent == ROOT_DIR


def test_cost_review_is_derived_not_hardcoded() -> None:
    # tau (0.067 h) * wage (INR 600/h) ~= INR 40. 07-math.md §5.
    assert 39.0 < COST_REVIEW_INR < 41.0


def test_every_script_accepts_seed() -> None:
    parser = base_parser("smoke")
    assert parser.parse_args([]).seed == SEED
    assert parser.parse_args(["--seed", "7"]).seed == 7


def test_seed_everything_is_reproducible() -> None:
    a = seed_everything(123).normal(size=5)
    b = seed_everything(123).normal(size=5)
    assert (a == b).all()
