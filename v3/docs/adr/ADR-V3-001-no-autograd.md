<!-- HEAD
FILE:     docs/adr/ADR-V3-001-no-autograd.md
PHASE:    architecture decision record
UPDATED:  2026-09-01
STATUS:   accepted — standing, and revisitable by the process in §Reversal
SUMMARY:  No autograd framework enters this tree. torch and transformers stay rejected,
          which gates Rung 7's gated-attention upgrade (T-0131) and Rung 8's neural
          conditional intensity (T-0132) as `conditional` rather than deferred.
OPEN:     Nothing. This file records a decision already made and already relied upon; it
          did not previously exist as a file, which is the defect it closes.
-->

# ADR-V3-001 — No autograd framework

## Status

**Accepted.** Standing since the v2 charter; recorded as a file for the first time in
cycle 4.

**This ADR is written to close a documentation defect, not to make a new decision.** It is
cited in two places that a reader can already reach — `src/rakshak/models/rung5_mil.py:9`
("ADR-V3-001 holds: no torch, so this is pooling, not attention") and
`configs/rung_roster.yaml`, which gates two rungs on it and records against each the honest
note that ``ADR-V3-001 has no file in this tree — `grep -rn ADR-V3-001` returns nothing``.
A constraint that shapes the ladder while existing only as folklore is not auditable, and a
panel reader who follows the citation and finds nothing is entitled to discount everything
downstream of it. The decision below is the one that was already being enforced.

## Context

Rakshak scores every cleared merchant, every day, under an analyst-capacity budget. The
build constraints that bear on this decision:

- **CPU-only, four cores, no GPU is available and none will be.** This is not a preference
  to be traded off; it is the machine.
- **A ~2-day solo build window** with the evaluation harness frozen before any model is
  written. Time spent on framework mechanics is time not spent on the ladder.
- **Labels, not compute, are the binding constraint.** At the cycle-4 day-239 boundary there
  are roughly 234 trainable positive merchants against 40,000; the validation fold carries
  7–14 merchants whose drift onset falls inside the scoring window. Rung 2 trains in 7.6 s
  and its artefact is 0.489 MB, against NFR budgets of 20 minutes and 20 MB. **Compute was
  never the constraint, and §8.10 of `LIMITATIONS.md` says so with the numbers.**
- **`make all` must pass from a clean `git clone` on a fresh environment**, enforced in CI.
  v1's single largest disqualification risk was a pipeline that did not reproduce on a clean
  checkout.
- **Permissive licences only.**

Two ladder entries want autograd and are blocked by this:

| entry | ticket | what it wants |
|---|---|---|
| Gated-attention MIL (Ilse et al., ICML 2018) as the Rung 5 upgrade | T-0131 (#65) | a learned attention pooling over payer capsules |
| Neural conditional intensity for Rung 8 | T-0132 (#66) | a neural temporal point process |

## Decision

**No autograd framework is introduced into this tree.** `torch`, `transformers`, and any
library that pulls an autograd runtime as a hard dependency stay rejected. Rungs are built
on `numpy`, `scipy`, `scikit-learn` and `lightgbm`, all already pinned.

Consequently T-0131 and T-0132 are **`conditional`**, not `deferred` — the distinction is
load-bearing and is why the roster uses that word. A deferred rung is one nobody got to. A
conditional rung is one whose blocker is named, written down, and reversible by a stated
process. The roster must continue to carry both as visible negative entries rather than
dropping them, so that the ladder records what was ruled out and on what grounds.

**A method is not rejected for being neural. It is rejected for needing gradients through a
framework this build cannot carry.** Hand-written numerical methods are in scope and are
used: Rung 5 is noisy-OR / log-sum-exp pooling written in plain numpy, and Rung 7a is a
hand-written HSMM-NB inference core. Neither needed autograd, and both were built.

## Consequences

**Accepted:**

- Attention-weighted pooling is unavailable; Rung 5 uses fixed noisy-OR / LSE pooling
  instead. `rung5_mil.py` states this at the top of the module, which is the right place.
- A neural conditional intensity is unavailable; Rung 8 has no admissible implementation
  under this ADR and is not on the scored ladder.
- Any cycle-4 survey recommendation requiring autograd is **GATED, not dropped**. A survey
  that silently omits such methods hides the constraint; one that names them and marks them
  gated leaves the decision visible and revisitable. This is the required treatment.

**Gained:**

- The dependency surface stays small and permissively licensed, so the clean-clone CI job
  stays fast and green — the K-5 risk (`make eval` does not reproduce on a clean clone) is
  the one that most nearly disqualified v1.
- Every rung on the ladder is readable end to end without a framework's abstractions in the
  way, which matters more than usual here because the project's deliverable is an argument
  about measurement, and an unreadable rung cannot participate in that argument.
- No GPU-shaped hole opens between the development machine and the stated deployment target.

**Costs, stated plainly:**

- The ladder cannot answer whether learned attention pooling beats fixed pooling on this
  problem. That is a real unanswered question and it should be reported as unanswered rather
  than as answered in the negative.
- Should the label constraint ever be relieved — it is currently the binding one — the case
  for reversal strengthens, because the argument below rests partly on 234 positives.

## Alternatives rejected

**Adopt torch on CPU.** It runs, but it buys nothing the constraint that actually binds. With
~234 trainable positives, a gated-attention layer's extra parameters are fitted on a sample
that cannot support them, and Rung 5's plain pooling already scored PR-AUC 0.7176 — *below*
Rung 2's 0.8357, so the pooling family was losing on this data before attention entered the
question. Adding autograd to a family that is behind is spending the scarcest resource in the
build on the least promising branch. It also enlarges the clean-clone surface materially.

**A lighter autograd (JAX, tinygrad, autograd).** Same category, smaller blast radius, same
answer for the same reason: the blocker is sample size, not framework weight. Adopting a
smaller framework would also weaken the constraint's clarity without changing any result.

**Hand-written backpropagation for one layer.** Feasible, and consistent with Rung 7a's
hand-written HSMM. Rejected on schedule rather than on principle: a bespoke gradient
implementation needs its own correctness evidence (a finite-difference gradient check at
minimum) and would consume the window that the geometry fix and the ladder rescore need. **If
this ADR is ever revisited, this is the alternative to revisit first** — it keeps the
dependency surface intact and the cost is bounded and known.

## Reversal

This ADR is reversed by a written amendment that states, before any code is written:

1. Which specific rung the framework is for, named.
2. The numeric adoption gate that rung must clear, declared in advance.
3. Evidence that the label constraint no longer binds — concretely, that the method's
   parameter count is supportable by the available positives.
4. A licence check on the framework and its transitive dependencies.
5. Confirmation that the clean-clone CI job still passes inside its budget.

GitHub #51 made the recommendation — *"The torch decision is made once, in the open, and the
recommendation is no"* — and this file is that decision in the open. It is not permanent, and
the process above is what makes it a decision rather than a habit.

## Housekeeping this ADR discharges

`configs/rung_roster.yaml` carries a `known_gap` and two per-rung `gap` fields, all saying
the same thing: the ADR has no file. **They are now stale in the good direction**, and the
roster should be updated to cite this path when it is next regenerated. The `gap` fields are
kept honest by pointing at a file that exists, not by being deleted.
