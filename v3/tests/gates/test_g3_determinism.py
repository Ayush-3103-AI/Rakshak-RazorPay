"""G3 — determinism. Two clean runs at the same seed produce identical output.

**Blocking.** If this is RED, nothing proceeds: an unseeded RNG anywhere in the generator
means the dataset a model trained on is not the dataset the report describes, and every
number downstream is unreproducible in a way no later test can detect.

CLAUDE.md's rule — every stochastic function takes ``rng: np.random.Generator``, no
module-level RNG, no bare ``np.random.*`` — is what makes this gate pass. This file tests
the property *and* the rule, because the property can hold while the rule is broken: a
bare ``np.random.random()`` reached from a process whose global seed happens to be fixed
produces identical runs right up until something else consumes a draw first.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from gates_report import green_if, scenario

from rakshak.generator.engine import generate

GENERATOR_DIR = Path(__file__).resolve().parents[2] / "src" / "rakshak" / "generator"
#: Small on purpose: determinism is a property of the code, not of the population size,
#: and this gate runs the generator twice.
DETERMINISM_MERCHANTS = 400
DETERMINISM_SEED = 1234


def test_g3_two_runs_at_the_same_seed_are_identical() -> None:
    config = scenario(n_merchants=DETERMINISM_MERCHANTS)
    first = generate(config, np.random.default_rng(DETERMINISM_SEED)).sha256()
    second = generate(config, np.random.default_rng(DETERMINISM_SEED)).sha256()
    ok = green_if(
        "G3 determinism",
        first == second,
        f"sha256 {first[:16]}... == {second[:16]}...",
        f"{DETERMINISM_MERCHANTS} merchants, seed {DETERMINISM_SEED}, all five tables",
    )
    assert ok, f"same seed produced different data: {first} vs {second}"


def test_g3_different_seeds_produce_different_data() -> None:
    """The control. A hash that returned a constant would pass the test above."""
    config = scenario(n_merchants=DETERMINISM_MERCHANTS)
    first = generate(config, np.random.default_rng(DETERMINISM_SEED)).sha256()
    other = generate(config, np.random.default_rng(DETERMINISM_SEED + 1)).sha256()
    assert first != other


def test_g3_no_module_level_rng_in_the_generator() -> None:
    """AST scan for ``np.random.<anything>`` other than a generator construction.

    Catches the cause rather than one of its symptoms.
    """
    offenders: list[str] = []
    for path in sorted(GENERATOR_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "random"
                and isinstance(value.value, ast.Name)
                and value.value.id in {"np", "numpy"}
                and node.attr not in {"default_rng", "Generator"}
            ):
                offenders.append(f"{path.name}:{node.lineno} np.random.{node.attr}")
    ok = green_if(
        "G3b no-global-rng",
        not offenders,
        f"{len(offenders)} bare np.random.* call(s) in src/rakshak/generator/",
        "; ".join(offenders) if offenders else "every stochastic function takes rng",
    )
    assert ok, f"bare np.random usage: {offenders}"
