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
  **Resolved 2026-08-30 (T-0020): `pymoo` was removed from `pyproject.toml`.** Nothing imported
  it. The obligation above is unchanged and still undischarged — only the dependency manifest now
  agrees with what was actually built.
* **The decision itself still stands** if the work is ever picked up: three objectives, NSGA-II,
  and the grid-search baseline is not optional.
* **What shipped instead** is a single-threshold Bayes Minimum Risk policy under a hard capacity
  constraint (ADR-0005), with sensitivity to the cost asymmetry swept and reported in
  `results/sensitivity.md`. That is a weaker claim than a Pareto frontier and must be described
  as one.

## Addendum — 2026-08-30 (T-0023, drift-detection literature survey)

**The rationale in "Context" above is wrong at exactly three objectives. The decision it
supports is unchanged; the reason given for it is not.**

Zheng & Doerr, *Runtime Analysis for the NSGA-II: Proving, Quantifying, and Explaining the
Inefficiency For Many Objectives* (arXiv:2211.13084; IEEE TEVC; GECCO 2024) prove that on the
m-objective generalisation of the OneMinMax benchmark — where every solution is Pareto optimal —
NSGA-II "cannot compute the full Pareto front ... in sub-exponential time when the number of
objectives is at least three", even with large population sizes. The stated mechanism is the
crowding distance itself: "in the computation of the crowding distance, the different objectives
are regarded independently", which is harmless at two objectives and breaks beyond two. Verified
at the arXiv abstract on 2026-08-30.

"Context" ¶2 asserts that reference-direction machinery "begins to matter at four or more
objectives". **Three is the first failing case, not the last safe one.** The same claim is
repeated at `CLAUDE.md:69` and `CLAUDE.md:81` and is wrong in both places.

**What changes:**

* The rejection of NSGA-III is **not** reversed. NSGA-III is a many-objective algorithm whose own
  runtime analyses target four or more objectives; nothing in the surveyed literature recommends
  it at three for a low-dimensional threshold search. Reversing to it would trade one
  under-evidenced choice for another.
* **Option (c), the uncoupled per-segment grid search, is promoted from mandatory baseline to
  preferred default.** If NSGA-II's diversity mechanism is provably degraded at three objectives,
  a dominance result for the coupled solution would have been the surprising outcome, not the
  expected one.
* **`pymoo` should be dropped from `pyproject.toml` at T-0020**, and the reason recorded as *the
  rationale did not survive review*, not merely *the work was cut*. The Consequences section above
  already flags the dependency as a defect to resolve before freeze; this addendum removes the
  argument for keeping it.

**What does not change:** no shipped number. T-0009 was cut on 2026-08-28 and neither algorithm
was ever built, so the repo has never made a Pareto-frontier claim. The obligation recorded in
"Decision" — that a GA must dominate the grid search in hypervolume or be deleted — remains
undischarged and is now, if anything, easier to discharge in the grid search's favour.

Source: `project-context/15-lit-survey-drift-detection.md` §Q3, ADR-0004.
