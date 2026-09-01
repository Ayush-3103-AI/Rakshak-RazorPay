<!-- HEAD
FILE:     project-context/12-spec-cycle4.md
PHASE:    spec (cycle 4)
UPDATED:  2026-09-01
STATUS:   ready-for-agent
SUMMARY:  Cycle 4. Restores time-to-detection as a measurable quantity by moving the data,
          not the harness; re-surveys three literatures against the two failures the
          cycle-3 ladder recorded; adopts at most one new rung from that survey.
OPEN:     The rung's identity is deliberately unnamed here. The survey names it, and the
          cycle-4 pre-registration fixes it before any rung code exists.
-->

# Spec — Cycle 4: measurable latency, a three-literature re-survey, and one new rung

## Problem Statement

The cycle-3 ladder recorded two failures. One of them is not a failure at all, and nobody
noticed because it was reported as a number rather than as an impossibility.

**First, time-to-detection was never measurable.** `detection_rate_d7`, `detection_rate_d14`
and `detection_rate_d30` read `0.000` for every rung *and every floor* on the cycle-3
validation split. `LIMITATIONS.md` §8.7 reports this as "nothing detects anything quickly."
The truer statement is that nothing *could*. Drift onsets are confined by the scenario config
to days 30-240 and the realised maximum is day 217. The validation window opens on day 240
and the test window on day 300. Time-to-detection is measured from a merchant's true onset,
so the earliest achievable TTD for any merchant is `240 - onset`: minimum 23 days, median
131.5 days.

- **d7**: 0 of 294 fraud merchants could have scored. It needs onset >= 233.
- **d14**: 0 of 294 could have scored. It needs onset >= 226.
- **d30**: 4 of 294 could have scored platform-wide, before the 15.2% merchant-fold split
  and the censoring filter take their cut.

An oracle that alerts on every merchant on the first day of the window scores exactly 0.000
on d7 and d14. The reported `ttd_median_days` of 162.0 is the distance from onset to the day
the window opened; it is a property of the split geometry and carries no information about
any model. Charter §2 makes TTD an equal-standing win condition, and the ladder has never
been able to render a verdict on it.

**Second, every rung is FLOOR-FAIL on savings against `volume_rank`.** The floor — alert the
K largest merchants by pre-window GMV, the same merchants every day, week-over-week alert
Jaccard exactly 1.000 — scores 0.6016. The best rung scores 0.4348. A rung ranking at PR-AUC
0.836 with precision@K 0.864 loses on rupees to a size ranking that ranks at PR-AUC 0.217.

These two failures are plausibly the same failure. If the scoring window contains no onsets,
then no drift occurs inside it, the fraud population is stationary across the whole window,
and a static size ranking is the *correct* model for that regime. The premise the project
was built on — catch a merchant as it drifts — has never had a window in which to happen.

**Consequently, no modelling work can be evaluated.** Multiple-instance learning, conformal
risk control, hidden semi-Markov models, Hawkes processes and neural intensities would every
one of them score d7 = 0.000 on this data. Any method chosen to improve latency would be
chosen against a metric that cannot move.

## Solution

Three changes, strictly in this order, with the ordering itself load-bearing.

**1. Move the data, not the harness.** Widen the generator's onset window to span the full
horizon so validation and test windows contain fresh onsets, and decouple label resolution
from the transaction horizon so those late onsets still resolve rather than censoring. This
makes TTD a quantity that discriminates between rungs for the first time.

The eval harness is **not modified**. `time_to_detection` and `detection_rates` were never
the defect — they are correct as written and start discriminating the moment the data
contains in-window onsets. The enforced `eval_module_sha256` is therefore byte-identical
between cycle 3 and cycle 4: the same hash-verifiable harness that scored the failing ladder
scores the new one. This is the strongest claim available about the fix and it costs nothing.

**2. Re-survey three literatures** against the two recorded failures: quickest change
detection for the latency objective; delayed-feedback and positive-unlabelled learning for
the label constraint; and the cost/decision layer for the `volume_rank` floor-fail. The
survey ends in a ranked recommendation with an ADR stub per literature.

**3. Adopt at most one new rung**, whichever the survey ranks first, pre-registered by name
and by adoption gate before a line of its code exists.

The whole ladder is rescored on cycle-4 data. A ladder with mixed-provenance rows is not a
ladder.

## User Stories

1. As a panel reviewer, I want to know whether a reported metric of zero means the model
   failed or means the metric could not fire, so that I can tell a negative result from a
   broken measurement.
2. As a panel reviewer, I want the arithmetic behind an impossibility claim shown to me, so
   that I can check it rather than take it on trust.
3. As a panel reviewer, I want to know that the evaluation harness did not change when the
   result changed, so that I can rule out the harness as the cause of the improvement.
4. As a panel reviewer, I want the hash of the eval module to be identical across the two
   cycles I am comparing, so that "we only moved the data" is verifiable and not a claim.
5. As a Head of Risk Operations, I want to know how many days after a merchant starts going
   bad the system actually flags it, so that I can size the chargeback exposure I am still
   carrying.
6. As a Head of Risk Operations, I want a detection-latency number that a naive floor cannot
   trivially match, so that I know the model is contributing something.
7. As a Head of Risk Operations, I want to know how many honest merchants are held per
   thousand, so that I can price the false-positive cost of running this.
8. As a Head of Risk Operations, I want to know why a size ranking beat every model on
   rupees, so that I can judge whether to just deploy the size ranking.
9. As a Head of Risk Operations, I want alerts ranked by what they save rather than by how
   confident the model is, so that my analysts spend a fixed budget of hours where the money
   is.
10. As a Head of Risk Operations, I want the alert budget respected as a hard ceiling rather
    than a quota, so that a quiet day does not manufacture alerts that cost more than they
    save.
11. As a builder, I want the onset window to span the horizon, so that every split contains
    merchants that turn bad inside it.
12. As a builder, I want label resolution decoupled from the transaction horizon, so that a
    merchant turning bad late in the simulation still resolves instead of censoring.
13. As a builder, I want the per-typology onset windows rescaled proportionally rather than
    flattened, so that no typology is made easier or harder relative to another.
14. As a builder, I want prevalence held at the BAF-native rate, so that the external anchor
    remains comparable and AP-06 is not reopened.
15. As a builder, I want the merchant count raised rather than the fraud rate, so that the
    evaluation denominator grows without distorting the prevalence claim.
16. As a builder, I want onsets drawn uniformly across the horizon rather than stratified
    into the evaluation windows, so that no reviewer can ask whether the evaluation window
    was enriched to make the metric measurable.
17. As a builder, I want the expected yield of the regeneration estimated before I run it,
    so that I find out the denominator is too small in a Monte Carlo rather than after a
    rescore.
18. As a builder, I want to be told the statistical power of the new TTD denominator, so
    that I do not read a difference between two rungs that the sample cannot support.
19. As a builder, I want the cycle-3 ladder tagged immutable before anything is regenerated,
    so that invalidating it is reversible.
20. As a builder, I want every rung rescored on the same cycle-4 data, so that no row on the
    ladder has different provenance from its neighbours.
21. As a builder, I want the capacity budget recomputed from the standing per-10,000 rate
    rather than carried over, so that K follows the population by rule instead of by habit.
22. As a builder, I want the floors rescored at the new K too, so that `volume_rank` is
    beaten at the budget the rungs actually face.
23. As a survey reader, I want the latency literature searched for methods whose objective
    function *is* detection delay, so that the ladder stops optimising a classification
    score and inheriting latency as a side effect.
24. As a survey reader, I want the multi-stream branch covered specifically, so that "which
    of N merchants changed, under one global false-alarm budget" is addressed as the stated
    problem rather than as N independent problems.
25. As a survey reader, I want methods that need no labels evaluated on their own merits, so
    that the binding label constraint stops capping every candidate.
26. As a survey reader, I want the delayed-feedback literature checked against this
    generator's actual delay distribution, so that a method is adopted because it matches
    the mechanism and not because it is fashionable.
27. As a survey reader, I want positive-unlabelled formulations assessed, so that censored
    merchants stop being silently treated as negatives.
28. As a survey reader, I want the cost and decision-layer literature searched for why a
    well-ranking detector loses on a loss-weighted score, so that the floor-fail gets an
    explanation and not just a report.
29. As a survey reader, I want every recommended library licence-checked, so that a
    permissive-licence-only constraint is not breached by an appendix.
30. As a survey reader, I want each candidate assessed against a four-core CPU budget, so
    that nothing is recommended that cannot be run here.
31. As a survey reader, I want methods requiring autograd flagged as gated rather than
    silently dropped, so that the standing rejection is visible and revisitable.
32. As a survey reader, I want a ranked recommendation table with an explicit first place, so
    that the implementation step has one unambiguous input.
33. As a survey reader, I want an ADR stub per literature, so that the reasoning survives the
    sprint that produced it.
34. As a survey reader, I want the places where the literature is thin named, so that the
    absence of a citation is not mistaken for the absence of a problem.
35. As a survey reader, I want a stated contrarian view, so that the recommendation has been
    argued against at least once before it is built.
36. As a pre-registration reader, I want the new rung named before its code exists, so that
    it cannot be chosen after seeing which candidate scored well.
37. As a pre-registration reader, I want its adoption gate stated numerically in advance, so
    that "it worked" is decided by a threshold rather than by narrative.
38. As a pre-registration reader, I want the seed list fixed before any model trains, so that
    a favourable seed cannot be selected afterwards.
39. As a pre-registration reader, I want the rule that conditionally opens the test split
    written as a number before validation is read, so that it is a stopping rule and not
    selective reporting.
40. As a pre-registration reader, I want the conditionality of the test-split opening
    disclosed in the limitations, so that a reader knows a held-out number's existence was
    itself contingent.
41. As a builder, I want the lock sealed before the data is regenerated, so that the
    ordering claim holds without qualification.
42. As a builder, I want the previous lock recorded as superseded rather than deleted, so
    that the lock history remains auditable.
43. As a builder, I want the test split to stay shut unless its pre-registered gate is met,
    so that the one-way door is not spent on a rung that did not earn it.
44. As a builder, I want the new rung to enter through the existing rung interface, so that
    the scoring path gains no new seam.
45. As a builder, I want the new rung to reach the capacity layer through the existing
    decision-policy seam, so that the locked selector is wrapped rather than edited.
46. As a builder, I want the new rung to add no dependency, so that the permissive-licence
    and no-autograd constraints hold without a new argument.
47. As a builder, I want a rung that is cut or not adopted to stay in the tree as a negative
    result, so that the ladder records what was tried and lost.
48. As a builder, I want any rung that fails its gate reported as failing rather than tuned,
    so that the kill criteria mean something.
49. As a report reader, I want the cycle-3 numbers preserved beside the cycle-4 ones, so that
    the effect of the geometry fix is visible as a before and after.
50. As a report reader, I want to be told which of the two failures the new rung addressed
    and which it did not, so that one improvement does not imply the other.
51. As a report reader, I want the new TTD denominator's size stated wherever a TTD rate is
    quoted, so that a rate over fourteen merchants is never read as a rate over three hundred.
52. As a report reader, I want per-typology recall reported separately, so that an easy
    typology cannot hide a hard one behind an aggregate.
53. As a report reader, I want savings quoted beside PR-AUC and beside the floor, so that no
    savings number is read as a claim about detection on its own.
54. As a maintainer, I want the ADR that the roster and a rung module both cite to exist as a
    file, so that a standing architectural constraint is readable rather than folklore.
55. As a maintainer, I want the rung roster updated to reflect what cycle 4 actually scored,
    so that a cut or deferred rung is named as such rather than being invisible.
56. As a maintainer, I want the stale deferral list corrected, so that project state as
    rendered to judges is not contradicted by the artefacts beside it.

## Implementation Decisions

### The seams

Four seams, chosen to be as few and as high as possible. One of them is deliberately empty.

**Seam 1 — the scenario config.** Population size, onset window, per-typology onset bounds
and the new label-resolution horizon are all config values. No code changes for any of them.

**Seam 2 — the generator's label emitter.** Censoring is currently decided by comparing a
merchant's `label_available_at` against the simulation end. It will instead compare against a
label-resolution horizon that defaults to the simulation end when unset. This is the only
generator code change in the cycle and it is one line and backward-compatible.

**Seam 3 — the rung module interface.** The new rung is another rung module producing a score
vector through the path the existing supervised rungs already use, and reaches the capacity
layer through the decision-policy seam introduced for conformal risk control. No new seam.

**Seam 4 — the eval package. Deliberately untouched.** The metrics were never wrong. Not
editing them is what makes the enforced hash identical across cycles, and that identity is a
deliverable of this spec, not an accident of it.

### Dataset

| decision | value | rationale |
|---|---|---|
| merchant count | 20,000 -> **40,000** | Buys evaluation denominator without touching prevalence. |
| horizon | 365 days, unchanged | Lengthening it spreads a fixed number of onsets thinner and *reduces* trainable positives; measured, not assumed. |
| prevalence | **1.47%, held** | BAF-native. Raising it would buy the same denominator by reopening AP-06. |
| onset window | days 30-240 -> **days 30-364** | Validation and test windows gain in-window onsets. |
| onset placement | **uniform**, not stratified | A stratified schedule would over-represent onsets in the evaluation windows by ~1.5x. Declared enrichment is defensible but invites the question; uniform does not. Costs ~4 merchants of denominator. |
| label resolution horizon | **day 500** | Transactions still end at day 364. Censoring is an artefact of where simulation stops, not a fact about the world; resolving analytically past the transaction horizon affects eval-side hindsight ground truth only. |
| split boundaries | **unchanged** | train 0-239, val 240-299, test 300-364. |
| seeds | **unchanged** | The five already declared. |
| capacity K | recomputed by the standing per-10,000 rate | Doubling the population takes the validation budget from 15 to approximately 30. K follows the rule, not the previous number. |

Per-typology onset bounds are rescaled by the affine map `[30, 240] -> [30, 364]`, preserving
each typology's original relative position and spread so that none is made easier or harder
relative to another — the same operation performed by hand when the current values were set.
This table is the decision and is reproduced because prose cannot carry it precisely:

| typology | current | rescaled |
|---|---|---|
| R1 | 59-189 | 76-283 |
| R2 | 30-131 | 30-191 |
| R3 | 44-218 | 52-329 |
| R4 | 37-189 | 41-283 |
| R5 | 37-204 | 41-307 |
| R6 | 88-240 | 122-364 |
| R7 | 30-218 | 30-329 |
| R8 | 37-218 | 41-329 |
| R9 | 30-160 | 30-237 |

**Expected yield, from a Monte Carlo over the real label pipeline** (`Exp(21) + U(45,120)`,
15% unreported, 15.2% validation merchant fold): approximately **134 trainable positives at
the day-239 boundary, 14 in-window onsets in the validation fold, 13 in the test fold.**

**The power ceiling is a decision, not a caveat.** A detection rate over ~14 merchants
carries roughly a ±13 pp standard error. TTD becomes measurable and fair; it does not become
well-powered. Two rungs differing by less than roughly 25 pp will not be separable, and the
report must say so wherever a TTD rate appears.

### Protocol

Ordering is load-bearing and is itself part of the spec.

1. The cycle-3 ladder is tagged immutable before anything is regenerated.
2. The survey is written. **No code is written during the survey.**
3. Cycle 4 is pre-registered: the regeneration parameters, the seeds, the identity of the one
   new rung, its numeric adoption gate, and the numeric validation gate that conditionally
   opens the test split.
4. The cycle-4 lock is sealed, recording the previous lock as superseded. The enforced
   eval-module hash is expected to be **unchanged**; a change means the eval package was
   edited and the claim in this spec has been broken.
5. The dataset is regenerated.
6. The full ladder — every floor and every rung — is rescored.
7. The new rung is implemented and scored.
8. The test split opens **once, and only if** the pre-registered validation gate is met. Its
   conditionality is disclosed wherever a test number is reported.

A rung whose code lands before step 4 is post-lock and ineligible for adoption. This is why
the survey cannot overlap the implementation.

### Method selection

The rung's identity is **deliberately not fixed by this spec.** The survey selects it and the
pre-registration fixes it. Constraints the survey must respect, so that its first-place
recommendation is implementable as written:

- No autograd. `torch` and `transformers` remain rejected; the ADR asserting this is cited in
  two places and does not exist as a file, and this cycle writes it.
- Permissive licences only.
- Four CPU cores, and the full evaluation must stay inside its runtime budget.
- No dependency added without a licence check and a logbook line.
- The method must enter through the existing rung interface and must not require an edit to
  the locked eval package.

The survey should weight, and be explicit that it is weighting, methods whose objective
function is detection delay rather than classification accuracy, and methods that do not
consume labels — because labels, not compute, are this project's binding constraint, and a
method that sidesteps that constraint is on a different axis from everything already on the
ladder.

## Testing Decisions

A good test here asserts an externally observable property of the data or of a scored result.
It does not assert that a function was called, and it does not restate a constant from the
config. The existing suite's most valuable tests are of this shape — a geometry check that
fails at config load, a parity check between two runners, a refusal to compute metrics above
capacity — and the new ones should match them.

**The geometry assertion is the important new test, and it is the one that would have caught
this.** It asserts that each evaluated split contains a minimum number of merchants whose
onset falls inside that split's own day range. It fails on the cycle-3 config. This is prior
art shaped like the existing config-load geometry checks, which already fail a config that
starves the test split of labelled positives — that check existed and passed, because it
counted labelled positives rather than in-window onsets, which is precisely the gap.

Also tested:

- **Config load** rejects per-typology onset bounds falling outside the population window,
  as it does today. The rescaled values must satisfy it.
- **Label resolution** asserts that a merchant onsetting after the transaction horizon's
  final scoring day but before the resolution horizon resolves rather than censoring, and
  that the horizon defaults to the previous behaviour when unset — so the change is provably
  backward-compatible.
- **The invariant** that `label_available_at > label_event_at >= drift_onset_at` continues to
  hold under the new horizon.
- **The existing floor** on labelled positives per seed in the test split continues to hold;
  expected yield is comfortably above it, and this should be confirmed rather than assumed.
- **Detection metrics** are asserted to be capable of firing: on regenerated data, a policy
  that alerts on every merchant every day must score a non-zero d30. On cycle-3 data this
  assertion fails, which is the point of writing it.
- **The lock** verifies, and the enforced eval-module hash is asserted equal to cycle 3's.
- **The new rung** gets whatever its method requires, plus the standard rung obligations:
  determinism under a fixed seed, at most K non-PASS actions on any day, and no access to
  labels or ground truth from inside the scoring path.

## Out of Scope

- **Editing the eval package.** Not a constraint to work around — a deliverable. If a
  candidate method requires a metric change, it is the wrong candidate for this cycle.
- **Any method requiring autograd**, including gated-attention multiple-instance learning and
  a neural conditional intensity. Both remain gated on an architectural decision this cycle
  documents but does not reverse.
- **More than one new rung.** The survey may rank several; one is adopted.
- **Retuning any existing rung.** They are rescored on new data, not re-tuned on it. A rung
  that gets worse gets reported as worse.
- **Fixing the generator's volume/fraud confound.** That `volume_rank` is a partially-informed
  detector on this generator is a known criticism of the generator and stays a stated
  limitation this cycle.
- **Raising prevalence** to buy statistical power.
- **The three registered features that are effectively constant** on generator output. Known,
  recorded, not this cycle's problem.
- **The dashboard, the video, and the root repository's pending deletions.**

## Further Notes

The finding in the Problem Statement is true today and depends on nothing in this spec. It
should be written into the limitations with its arithmetic regardless of whether any later
step lands. Under a track whose published bar is *honest metrics including false-positive
cost*, a project that caught its own headline win condition being unmeasurable by
construction, proved it, and fixed the measurement rather than the number, is making the
argument the track asked for.

The largest risk is schedule. The regeneration invalidates the cycle-3 ladder, so a run that
stops half-way leaves neither a complete cycle-3 story nor a complete cycle-4 one. Tagging
the cycle-3 ladder immutable first is what makes that recoverable, and it is step 1 for that
reason.

The second risk is that the novelty is downstream of the regeneration. If the regeneration
does not land, a latency-oriented rung has no latency to find and will score no better than
the floor it was built to beat.

The two failures may be one failure, and the report should be careful not to assert this. If
in-window onsets restore latency *and* the floor-fail closes, the natural reading is that
`volume_rank` was winning because the window was stationary. That is a hypothesis this cycle
can support but not establish, and it should be written as such.
