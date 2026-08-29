# ADR-0008 — Review capacity is expressed per 1000 merchants, not as an absolute figure

**Status:** Accepted — 2026-08-28 (T-0003b)
**Supersedes:** the absolute reading of `config.REVIEW_CAPACITY_HOURS`

## Context

FR-017 makes a fixed analyst-hour budget the constraint that the whole decision layer
exists to serve: pass / review / hold under scarce review capacity.

`config.REVIEW_CAPACITY_HOURS = 40.0` with `TAU_REVIEW_HOURS = 0.067` buys
`floor(40 / 0.067) = 597` reviews per decision period. Every split in the frozen
evaluation holds at most 300 merchants. The constraint was therefore slack everywhere:
the harness reviewed all 100 merchants of the validate split, `precision@K` collapsed to
prevalence (0.20 for every model, including `random`), gap-to-oracle measured cost-matrix
error rather than capacity, and all of T-0006's baselines would have tied exactly.
Flagged by T-0005; the numbers are in the T-0005 run of `results/summary.md`.

## Options considered

**(a) Express capacity per merchant population.** `REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS`,
derived against the size of the split being scored. One constant, no data change, and the
constraint becomes a property of the risk-ops team rather than of the dataset.

**(b) Keep 40 h absolute and grow the population to ~6000 merchants** so 597 slots are
~10% of the book. Costs generator runtime and eval runtime and pushes against NFR-004
(`make eval` under 15 minutes) for no analytical gain.

## Decision

(a). `REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS = 4.0` — about 60 reviews per 1000
merchants at `TAU_REVIEW_HOURS`, i.e. roughly 6% of the book per decision period, which is
a plausible load for a risk-ops desk. `eval/harness.py` derives
`capacity_hours = 4.0 * n_merchants / 1000` from the split it is scoring and passes it to
both `review_slots` and `review_knapsack_oracle`. `REVIEW_CAPACITY_HOURS` stays in
`config.py` for any caller that genuinely wants a fixed pool, but the harness does not
read it.

## Consequences

* On the 100-merchant validate split K falls from 100 to 5. The constraint binds: even
  the perfect-foresight knapsack oracle can only reach 5 of the 20 bad merchants.
* `precision@K` is now discriminative rather than pinned at prevalence, so T-0006's
  baselines can be told apart. Every reported `precision@K` must state K and the
  prevalence beside it.
* Savings and gap-to-oracle move. Numbers from any run before this ADR are not
  comparable to numbers after it.
* K depends on the split's population, so K differs between validate (100 merchants) and
  test (100). Any table that quotes `precision@K` must quote K with it.
* The absolute figure remains reachable and misleading. If a future caller imports
  `REVIEW_CAPACITY_HOURS` for an evaluation, this ADR is what it violates.
