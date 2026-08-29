# ADR-0004 — NSGA-II, not NSGA-III; and the grid-search obligation that comes with it

**Status:** Accepted (decision) — **but NOT IMPLEMENTED. T-0009 was cut on 2026-08-28.**
Decision taken in Phase 2; **written retrospectively on 2026-08-29** from `CLAUDE.md`'s stack
table, `07-math.md` §7, `08-pseudocode.md:257` and `11-tickets/T-0009.md`.
**Supersedes:** none.

## Context

`07-math.md` §7 poses policy optimisation as multi-objective over per-segment threshold pairs,
minimising three objectives — expected fraud loss, expected false-positive cost, and review
hours — subject to review hours ≤ B.

**Three objectives is not many-objective.** NSGA-III's reference-direction machinery exists to
maintain diversity when the Pareto front is high-dimensional, which begins to matter at four or
more objectives. Applying it at three adds machinery and a diversity mechanism that buys nothing
and invites the question "why?"

The second half of this decision matters more than the first. A genetic algorithm in a portfolio
submission is a resume line unless something measures whether it earned its place.

## Options considered

**(a) NSGA-III.** Wrong tool at three objectives.
**(b) NSGA-II** (pymoo ≥ 0.6.2, Apache-2.0, released 27 Jun 2026, actively maintained).
**(c) Uncoupled per-segment grid search** over threshold pairs. Trivial, fast, and — crucially —
it may be *good enough*, because segments may not actually interact through the shared budget as
much as the coupled formulation assumes.

## Decision

(b), **with (c) built as a mandatory baseline, not an optional one.**

**The obligation this ADR creates**, stated in `07-math.md` §7 and `08-pseudocode.md:257`: if
NSGA-II's coupled solution does not dominate the uncoupled grid search **in hypervolume**, then
NSGA-II is decoration — delete it and ship the grid search. This is the ablation that converts a
GA from a resume line into a measured result.

## Consequences

* **Neither was built.** T-0009 was cut in the 2026-08-28 re-plan, behind T-0006b (the HMM had no
  scorer) and the cost-matrix repair. `11-tickets/BOARD.md` records the cut; T-0011's ablation
  table carries the "NSGA vs grid-search" row struck and marked **not measured**, never zero and
  never silently absent.
* **So the obligation above is undischarged**, and that is the honest status. The repo must not
  claim a multi-objective frontier. `pymoo` remains in `pyproject.toml` as a declared dependency
  for work that did not happen — **that is a defect to resolve before freeze**: either remove it
  or state why it is there.
* **The decision itself still stands** if the work is ever picked up: three objectives, NSGA-II,
  and the grid-search baseline is not optional.
* **What shipped instead** is a single-threshold Bayes Minimum Risk policy under a hard capacity
  constraint (ADR-0005), with sensitivity to the cost asymmetry swept and reported in
  `results/sensitivity.md`. That is a weaker claim than a Pareto frontier and must be described
  as one.
