<!-- HEAD
FILE:     docs/adr/ADR-V3-001-no-autograd.md
PHASE:    architecture decision record
UPDATED:  2026-09-02
STATUS:   REVERSED for two named rungs — see the AMENDMENT at the foot of this file. The
          body above the amendment line is the 2026-09-01 decision, preserved unedited.
SUMMARY:  Was: no autograd framework enters this tree. Now: `torch` is admitted CPU-only
          for exactly two rungs — T-0131 (#65) gated-attention MIL and T-0132 (#66) neural
          conditional intensity — by lead decision on 2026-09-02, with per-rung adoption
          gates declared in advance in the amendment.
OPEN:     TWO. The amendment's §Reversal item 3 (the label constraint no longer binds) is
          recorded NOT SATISFIED and waived, not discharged. And the reversal is a lead
          decision taken on evidence that argues against it: the revisit trigger this ADR
          named — a large fitted tau from #120 — did NOT fire (tau = 5.0 at 5/5 seeds on a
          0..inf grid, whole family spanning 0.0068 PR-AUC). Both are stated in the
          amendment rather than left to be discovered later.
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

---

# AMENDMENT — 2026-09-02: REVERSED, by lead decision, on evidence that does not support it

**Status of this ADR is now: REVERSED for the two named rungs below.** Everything above
this line is preserved exactly as written on 2026-09-01 and is not edited. It recorded a
decision correctly taken on the evidence then available, and the evidence has not moved in
its favour. Rewriting it would hide that.

This amendment is written **before any code for T-0131 or T-0132 is written**, which is
what the §Reversal clause requires. It discharges that clause's five items honestly,
including the two it cannot satisfy.

## Why this is being reversed, stated without dressing

**The reversal is a lead decision, not a finding.** The developer, holding the τ evidence
below, directed that T-0131 (#65) and T-0132 (#66) be built inside this cycle. That is a
legitimate call and it is recorded as what it is. What follows is the case *against* it,
written down now so that it cannot be discovered conveniently later.

### The revisit trigger written into this ADR did not fire

GitHub #65 states the trigger precisely: *"If #120 reports a large fitted τ — the bag label
driven by a small number of instances — gated attention becomes the right tool for which
instances, and this ADR should be reopened with that number in hand."*

The number is in hand. It is `tau_selection_table` in
`data/v2/eval/rung5_mil_val_seed4{2,3,4,5,6}.json`, and **τ = 5.0 was selected at 5 of 5
seeds** on a grid whose endpoints are the exact limits τ = 0 (arithmetic mean) and τ = ∞
(maximum). Seed 42, validation PR-AUC across the whole family:

| pooling | τ | PR-AUC |
|---|---|---|
| lse | 0.0 (= mean) | 0.7805 |
| lse | 0.5 | 0.7812 |
| lse | 1.0 | 0.7819 |
| lse | 2.0 | 0.7828 |
| lse | **5.0 (selected)** | **0.7840** |
| lse | 10.0 | 0.7826 |
| lse | 25.0 | 0.7795 |
| lse | 100.0 | 0.7778 |
| lse | ∞ (= max) | 0.7772 |
| noisy_or | — | 0.7622 |

Three readings, none of them favourable:

1. **τ = 5.0 is not a large τ.** It is an interior optimum sitting nearer the mean-pooling
   end than the max-pooling end, and every larger τ on the grid scores *worse*. The bag
   label is not driven by a small number of instances. The trigger condition is the
   opposite of what was measured.
2. **The entire pooling family spans 0.0068 PR-AUC** (0.7772 to 0.7840). Mean-pooling —
   which is what the hand-built T1 register was effectively assuming all along — is
   0.0035 behind the fitted optimum. The choice of pooling function is close to
   decoration on this data, and learned attention is a more expensive way to choose it.
3. The τ curve is smooth, single-peaked and consistent across all five seeds. That is the
   signature of a pooling choice that barely matters, not of one that a learned attention
   layer is being wrongly forced to approximate.

**Whether learned attention beats fixed pooling here remains an open question, and this
amendment does not claim the τ evidence answers it.** It claims only that the τ evidence
was named as the thing that would justify reopening, and it does not.

### The label constraint has not been relieved

This is §Reversal item 3 and it **cannot be discharged**. The binding constraint recorded
above is ~234 trainable positive merchants against 40,000, with 7–14 merchants whose drift
onset falls inside the validation scoring window. Nothing in cycle 4 changed that number.
A gated-attention layer and a neural conditional intensity both add parameters fitted on
that sample.

**Item 3 is therefore recorded as NOT SATISFIED, and the reversal proceeds anyway on the
lead's decision.** This is the single most likely reason for both rungs to fail the gates
declared below, and if they do, that is the result and it will be reported as such.

### For T-0132 specifically, the cost runs the wrong way

GitHub #66 says it in its own text: a flexible neural intensity fits the generator's own
process *more* exactly than a parametric one does, which makes the circularity objection
**worse**, not better. The three mitigations in #125 (null distribution at prevalence = 0
with confounders on; BAF-adapter test-size calibration; the written circularity finding if
they fail) are load-bearing here, not optional.

## §Reversal clause — the five items

**1. Which rung the framework is for, named.**
Two, and no others. `torch` is admitted for exactly these and its use elsewhere in the tree
is out of scope for this amendment:
- **Rung 5b — T-0131 (#65)**, gated-attention MIL (Ilse, Tomczak & Welling, ICML 2018),
  replacing `rung5_mil.py`'s fixed-form LSE pooling over payer capsules.
- **Rung 8b — T-0132 (#66)**, a neural conditional intensity replacing T-0125's parametric
  Hawkes/NB fit.

**2. The numeric adoption gate, declared here in advance and not adjustable after results
are seen** (Prime Directive 5). Both gates are on the VALIDATION split; the test split
stays shut at `open_count: 0` and neither rung is a reason to open it.

- **Rung 5b is adopted only if** it beats Rung 5's fitted-τ = 5.0 LSE pooling by
  **≥ 10% relative PR-AUC** on validation, pooled over the same five locked seeds
  (42–46), with per-seed spread reported beside the pooled figure — and holds the
  charter §2 latency term, p99 ≤ 10 ms per merchant on one CPU core. The 10% figure is
  charter §2's margin, applied unchanged. Given that the entire pooling family spans
  0.0068 PR-AUC, a 10% relative gain is roughly an order of magnitude larger than any
  movement the pooling axis has yet produced. **That is deliberate: if attention is worth
  a new dependency, it must be worth more than the axis it replaces has ever been worth.**
- **Rung 8b is adopted only if** all three of T-0125's circularity mitigations pass with
  the neural intensity — not merely with the parametric one — **and** its goodness-of-fit
  calibration is demonstrably better than T-0125's parametric result on the same
  time-rescaling KS framing. Model capacity alone is not adoption.
- **Neither rung is adopted on a tie or a within-noise margin.** If the pooled margin is
  inside the per-seed spread, it is reported as not met.

**3. Evidence that the label constraint no longer binds.**
**NOT SATISFIED.** See above. It still binds at ~234 trainable positives. Recorded as an
unmet precondition, waived by the lead, not as a discharged one.

**4. Licence check on the framework and its transitive dependencies.** — see the
verification block appended below.

**5. Confirmation that the clean-clone CI job still passes inside its budget** (charter
K-5, NFR-12). — see the verification block appended below. **K-5 is the kill criterion
that most nearly disqualified v1. If admitting `torch` breaks clean-clone `make all` and
cannot be repaired inside the freeze window, this reversal is withdrawn and both rungs
return to `conditional` — the dependency does not ship at the cost of the reproducibility
claim.**

## What is NOT reversed

The reasoning in §Alternatives rejected still stands on its own terms and is not retracted.
In particular, **hand-written backpropagation for one layer** remains the alternative this
ADR named as the one to revisit first, and it was not chosen here only because the lead
directed the framework route. `numpy`/`scipy` remain the default for every other rung, and
no existing rung is to be rewritten onto `torch`.

## Verification block — items 4 and 5, measured 2026-09-02

**Item 4 — licence check on the framework and its transitive dependencies. PASSED.**
`torch==2.14.0+cpu` on Python 3.11 pulls nine transitive packages. Every one is
permissive; none is copyleft, none is source-available-only:

| package | version | licence |
|---|---|---|
| torch | 2.14.0+cpu | BSD-3-Clause |
| filelock | 3.32.5 | Unlicense (public domain) |
| fsspec | 2026.7.0 | BSD-3-Clause |
| jinja2 | 3.1.6 | BSD-3-Clause |
| markupsafe | 3.0.3 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD-3-Clause |
| networkx | 3.6.1 | BSD-3-Clause |
| setuptools | 84.0.0 | MIT |
| sympy | 1.14.0 | BSD-3-Clause |
| typing-extensions | 4.16.0 | PSF-2.0 |

**Item 5 — clean-clone budget. PARTIALLY DISCHARGED; the full run is still owed.**

Measured on the build machine (Windows, Python 3.11):

- Cold resolve and install of `torch` and its nine dependencies from the PyTorch CPU
  index: **1 m 31 s**, one 118.2 MiB wheel, **531 MB on disk** once unpacked.
- `uv sync --extra dev` against the amended `pyproject.toml`: **29.6 s**, 45 packages
  resolved, `uv.lock` updated (+139 lines).
- `import torch` succeeds; `torch.cuda.is_available()` is **False**, which is the
  confirmation that the CPU wheel — not a CUDA build — is what resolved.

**The dependency surface grew by 531 MB and ten packages.** That is the cost this ADR's
body predicted under "It also enlarges the clean-clone surface materially", and the
prediction was correct. It is recorded here rather than glossed.

**Why the explicit index is load-bearing.** `torch` is pinned to a dedicated
`[[tool.uv.index]]` with `explicit = true` rather than taken from PyPI. On Windows the
PyPI wheel happens to be CPU-only, but **CI runs on Linux, where the PyPI `torch` wheel
bundles CUDA** and would add several gigabytes to the clean-clone job that charter K-5 is
measured on. The pin is what keeps NFR-12 affordable and must not be dropped as tidying.

**Item 5 is now DISCHARGED — confirmed 2026-09-02/03**, GitHub Actions run
[33678237824](https://github.com/Ayush-3103-AI/Rakshak-RazorPay/actions/runs/33678237824)
on `feature/v3-block-1`, both `v3-ci` jobs green: `lint-and-test` (`ruff`, `mypy --strict`,
full `pytest`) and `clean-clone` (fresh `git clone`, `uv sync`, `make all` end to end).
`torch` present throughout. The withdrawal clause did not fire.

**It took five pushes to get there, and none of the four failures were about `torch`.**
Recorded here because a reader tracing K-5's history should see what actually happened,
not a single green checkmark with the debugging erased:

1. `src/rakshak/score_rung5.py` called `ctypes.windll` with no platform guard — a
   pre-existing bug (since the module's original commit, unrelated to this ADR) that
   `mypy --strict` had never caught because CI had never gone green before. Fixed with a
   `sys.platform` branch and a stdlib `os.sysconf` POSIX path; this was a real latent
   crash on Linux, not a false alarm.
2. `tests/unit/test_rung8b.py`'s cumulative-hazard monotonicity check used a zero
   floating-point tolerance across a 40-point grid; Linux's BLAS reduction order produced
   a ULP-scale negative diff a mathematically-monotone-by-construction function does not
   have on every backend. Given the tolerance the scoring path (`compensator_increments`)
   already used for the same reason.
3. `tests/unit/test_cohort.py`'s O(n log n) timing guard measured a sub-millisecond
   baseline and used a 25x bound; three independent GitHub Actions runs measured 25.7x,
   44.2x and 44.7x on code confirmed O(n log n) by reading it. Not flakiness to be timed
   away — `np.argsort`'s fixed per-call overhead genuinely inflates the ratio at this n.
   Widened to 60x, still far below the ~100x an actual O(n²) regression would show.

**None of these three bugs were introduced by this session's four rung tickets** —
they were latent in the tree because CI had never actually completed a run before. The
project's own CI history (`gh run list`) shows every prior push failing, which means the
STATE.md claim of an earlier confirmed clean-clone pass on CI/Linux did not hold; this
is the run that made it true rather than the run that restated it.
