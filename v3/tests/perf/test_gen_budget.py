"""NFR-10 — 10,000 merchants x 180 days generated in <= 3 minutes on 4 cores.

The one budget in this suite that is not about serving. It is here because the generator is
regenerated from seed on every clean clone (NFR-12), so its runtime is on the critical path
of ``make all`` and of anyone reproducing the results — and a dataset that takes an hour to
rebuild is a dataset nobody rebuilds, which is how a reproducibility claim quietly stops
being true.

Marked slow: it runs the real generator at the real population. There is no smaller version
of this test that means anything, because the number being asserted is the full run.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from perf_budgets import ROOT, assert_budget

from rakshak.generator.config import load_scenario
from rakshak.generator.engine import generate

GEN_BUDGET_S = 180.0
SCENARIO = ROOT / "configs" / "scenario_v2.yaml"


@pytest.mark.slow
@pytest.mark.skipif(not SCENARIO.exists(), reason=f"no scenario manifest at {SCENARIO}")
def test_generator_builds_the_full_population_inside_the_budget() -> None:
    scenario = load_scenario(SCENARIO)
    population = scenario.population
    assert (population.n_merchants, population.n_days) == (10_000, 180), (
        f"NFR-10's budget is quoted for 10,000 merchants x 180 days; the manifest says "
        f"{population.n_merchants} x {population.n_days}. Measuring a different population "
        f"against this budget would be measuring a different requirement."
    )

    started = time.perf_counter()
    data = generate(scenario, np.random.default_rng(42))
    elapsed = time.perf_counter() - started

    print(
        f"\nNFR-10 generator, {population.n_merchants:,} merchants x {population.n_days} days:"
        f"\n  {data.transactions.height:,} transactions in {elapsed:.1f} s"
    )
    assert data.transactions.height > 0, "the generator produced nothing; the timing is of a bug"
    assert_budget("NFR-10", "full generator run", elapsed, GEN_BUDGET_S, "s")
