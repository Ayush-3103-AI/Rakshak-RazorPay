"""NFR-10 — the generated dataset stays cheap enough to rebuild, at whatever population.

Quoted originally as **10,000 merchants x 180 days in <= 3 minutes on 4 cores**. That is a
rate, and this module now treats it as one: the budget is derived from the shipped
population at the same seconds-per-merchant-day, rather than pinned to a population the
manifest has twice moved away from.

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

#: NFR-10 as originally quoted: 180 s for 10,000 merchants x 180 days.
NFR10_BUDGET_S = 180.0
NFR10_MERCHANTS = 10_000
NFR10_DAYS = 180
#: Seconds per merchant-day implied by that quote. The budget is a RATE, not a constant.
GEN_BUDGET_S_PER_MERCHANT_DAY = NFR10_BUDGET_S / (NFR10_MERCHANTS * NFR10_DAYS)
SCENARIO = ROOT / "configs" / "scenario_v2.yaml"


def budget_for(n_merchants: int, n_days: int) -> float:
    """NFR-10's rate, applied to the population that actually ships.

    **This is an amendment, and it is written here rather than made silently.** The test
    previously asserted ``(n_merchants, n_days) == (10_000, 180)`` and failed outright
    otherwise, on the sound ground that measuring a different population against a fixed
    budget measures a different requirement. But the manifest moved to 20,000 x 365 in
    T-0101 and to 40,000 x 365 in cycle 4, so the assertion has been **red since T-0101** —
    and `perf` is a stage of `make all`, so the clean-clone job the board records as green
    has been failing at it for two cycles.

    Refusing to measure was right; refusing forever is not. NFR-10's intent is that the
    dataset stays cheap enough that people actually rebuild it — "a dataset that takes an
    hour to rebuild is a dataset nobody rebuilds, which is how a reproducibility claim
    quietly stops being true", in this module's own words. That intent is a rate. Holding
    the rate fixed and letting the budget follow the population preserves the requirement
    exactly; holding the number fixed abandons it the moment the population changes.

    At cycle 4's 40,000 x 365 the derived budget is ~1,460 s. Measured: 121,896,985
    transactions in 215.3 s, 6.78x inside it.
    """
    return GEN_BUDGET_S_PER_MERCHANT_DAY * n_merchants * n_days


@pytest.mark.slow
@pytest.mark.skipif(not SCENARIO.exists(), reason=f"no scenario manifest at {SCENARIO}")
def test_generator_builds_the_full_population_inside_the_budget() -> None:
    scenario = load_scenario(SCENARIO)
    population = scenario.population
    budget = budget_for(population.n_merchants, population.n_days)
    assert budget > 0.0

    started = time.perf_counter()
    data = generate(scenario, np.random.default_rng(42))
    elapsed = time.perf_counter() - started

    print(
        f"\nNFR-10 generator, {population.n_merchants:,} merchants x {population.n_days} days:"
        f"\n  {data.n_transactions:,} transactions in {elapsed:.1f} s"
        f"\n  budget {budget:.0f} s, derived from NFR-10's rate "
        f"({NFR10_BUDGET_S:.0f} s for {NFR10_MERCHANTS:,} x {NFR10_DAYS})"
    )
    # `n_transactions`, NOT `data.transactions`. The latter is a property that
    # materialises the whole table, which the generator deliberately stopped doing when
    # it moved to streaming merchant-contiguous blocks: at 40,000 x 365 that is 121.9M
    # rows and raises ArrayMemoryError before this assertion is reached. The count is
    # what this line wants, and the streaming field carries it for free. A timing test
    # that OOMs on the object it is timing is measuring the test, not the generator.
    assert data.n_transactions > 0, "the generator produced nothing; the timing is of a bug"
    assert_budget("NFR-10", "full generator run", elapsed, budget, "s")
