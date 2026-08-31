# LIMITATIONS — Rakshak v2

> Honest failures, with numbers. Prime Directive 6: *a rung that loses is a finding, not an
> embarrassment.* This file accumulates through the sprint and is finalised at T-152. Every
> failed rung, every fired kill criterion, and every cut feature is named here with its
> number and its reason.
>
> Nothing in this file is here because it was too awkward to fix elsewhere. Each entry names
> what was cut, why, and what it would take to un-cut it.

**Status:** accumulating · last updated 2026-08-31, after Lane B (T-120…T-122)

---

## 1. Features specified in the register but not built

Eight of the register's rows are not in `registry.ORDER`. The register is the contract
between the generator spec and the feature layer, so a row that is not built is a gap in
that contract and is named here rather than quietly dropped.

| Feature | Family | Why it was cut | What would un-cut it |
|---|---|---|---|
| `s_payout_freq_z` | F8 | **Structural.** `FeatureSpec.update(state, event)` takes a `Transaction`, and `assert_parity(spec, txns, profiles)` has no payout channel. There is no path for a `Payout` to reach a feature at all. | `update(state, event: Transaction \| Payout)` and a `payouts=()` argument on the parity harness — an amendment to the frozen `09-interfaces.md` §7. Declined 2026-08-31 to keep the Block-1 freeze intact. |
| `s_balance_drawdown` | F8 | Same. | Same. |
| `d_refund_latency_med` | F6 | **Rule 2 — online-computable or out.** The median latency between a capture and its refund requires the capture timestamp of an arbitrary past event, and the set of still-open captures is unbounded. It cannot be maintained in bounded state. | A `latency` (or original-capture-time) field carried on the refund event itself, in `schemas.py`. This is a permanent cut under the current schema, not a deferral. |
| `i_bin_hhi` | F3 | State budget. | The shared daily ring below. |
| `g_payer_hhi` | F4 | State budget. | The shared daily ring below. |
| `g_device_reuse_rate` | F4 | State budget. | The shared daily ring below. |
| `g_unique_payers_z` | F4 (T1) | **Ticket gap, not a technical cut.** T-120's scope names F1/F2/F5/F6/F7/F8/F9; T-122's names F2/F3/F4 *T2 rows only*. These two T1 rows fall between the two tickets and are in no ticket's scope. | A ticket. Flagged so the omission is not mistaken for a judgement about the feature. |
| `g_new_payer_ratio` | F4 (T1) | Same. | Same. |

**The shared daily ring.** Six T1 features keep private ring buffers over substantially the
same per-merchant daily series. One shared ring on `MerchantState` would return roughly 1 KB
and make the three budget-cut features affordable. It needs a slot in `state.py`, which
`09-interfaces.md` §8 freezes. Declined 2026-08-31 for the same reason as F8: reopening the
Block-1 contract mid-sprint costs more than the three features are worth, and the freeze is
itself a claim the project makes.

---

## 2. Failed and unmet acceptance clauses

### T-120 — the T1 state bar is not met
The ticket requires *total declared `state_bytes` for T1 < 1024*. **Measured: 1720 B across
24 T1 features.** The per-feature byte figures in `07-feature-register.md` are not physically
achievable as written — a trailing-30-day sum cannot be maintained in 16 B. The register's
budget column is an estimate that was never costed against an implementation.

Total declared state across all 28 features is **3968 B against NFR-04's 4096 B**, so the
*binding* budget is met; it is the T1-only sub-budget that is not.

### NFR-04 — measured state exceeds the budget, for a reason that is not the features
Declared state is 3968 B. **Measured `MerchantState.nbytes()` is 7091 B.** The gap is
approximately 120 B of pickle framing per `FeatureState` object across 28 objects. This is a
serialization problem, not a feature problem: the declarations are honest and the packing is
not. `MerchantState` needs a packed representation before NFR-04 can be asserted rather than
estimated. **Carried to T-150**, which owns the perf assertions.

### T-121 — the original clause was unreachable, and is preserved as a strict xfail
The ticket asked for `mean |residual| < 0.25` under confounder P2 at prevalence=0. Measured
against real generator output at both 3,000 and 10,000 merchants: mean |raw z| = 3.301, mean
|residual| = 3.117.

The null day is what settles it. **On a day with no confounder at all, mean |residual| was
already 0.558.** A z-score has unit scale by definition and E|N(0,1)| = 0.798, so subtracting
a cohort median cannot bring the mean absolute value below roughly 0.6 unless every merchant
within a cohort agrees to within 0.3σ, which no per-merchant z-score does. **A perfect
residual layer fails this clause and so does a useless one.**

The clause was amended on the board (dated, with the arithmetic) to a three-part mechanism
check. The original assertion is kept verbatim in `tests/unit/test_cohort.py` as
`xfail(strict=True)` rather than deleted. It was not weakened to go green.

**This is not a K-1 falsification.** K-1's verdict is rendered at T-142 on the Rung 3 vs
Rung 2 validation delta. Reporting a falsification caused by a mis-specified metric would be
a false claim about the hypothesis.

---

## 3. What the cohort-residual layer does and does not do

Measured, prevalence=0, confounder P2 (gateway outage) active, feature
`f_auth_fail_rate_z`, at 3,000 and 10,000 merchants:

| | mean \|z\| | mean \|residual\| | median z | median residual | alert rate at \|·\|>3 |
|---|---|---|---|---|---|
| **P2 day** | 3.301 | 3.117 | +1.790 | +0.005 | 0.390 → 0.231 |
| **null day** | 0.691 | 0.558 | −0.378 | −0.000 | 0.024 → 0.028 |

**It removes the common mode exactly.** Median z of +1.790 becomes a median residual of
+0.005. That is the mechanism working precisely as designed.

**It cuts confounder-driven alerts materially but not to zero** — 39.0% to 23.1% of the
population at a |·| > 3 threshold. A 41% relative reduction, and still a G5 catastrophe on
this feature alone.

**It cannot remove P2's per-merchant heterogeneity.** The generator scales each confounder
per merchant by `sqrt(target_fano / λ)` times persona sensitivity, so a six-hour outage hits
a five-transaction-a-day merchant far harder than a five-hundred-a-day one. A median removes
only what is shared, and most of P2's effect on this feature is not shared. Cohorting on
`gmv_decile` is the lever that would make more of it shared; at 3,000 merchants the backoff
chain rarely reaches the full key, and at 10,000 the improvement was marginal (mean
|residual| 3.135 vs 3.117).

**This is the honest framing for the writeup:** the residual layer is a partial defence
against platform-wide confounders, not a solution to them. Claiming otherwise would not
survive the first question from a risk operations panel.

---

## 4. Accepted deviations from spec

### `EVAL-LOCK.json` enforces one hash of three (eval-harness-spec §6)
§6 requires all three recorded hashes to hard-fail on mismatch. The harness was frozen while
the generator was still in flight — which `STATE.md` calls *preferable*, since a harness
frozen before the generator is harder to accuse of hindsight. Enforcing the generator and
config hashes would have hard-failed on the generator's next commit, someone would have added
`--force`, and within a day it would have been reflex. **A lock that is routinely overridden
is worse than no lock, because it still looks like evidence.**

`eval_module_sha256` is enforced — that is the actual claim. The other two are recorded as
freeze-time provenance and reported as drift, never silently. `verify_lock(strict=True)`
promotes all three once the generator freezes. The reasoning is written into the lock file's
own `enforcement_note`, readable by a reviewer who opens nothing else.

### `RungOutput`, `Truth` and `CostParams` live in `metrics.py`, not `schemas.py`
Against the "all types in `schemas.py`" convention. `schemas.py` froze at the end of Block 1
and a DESCEND to add three intra-package types would have cost more than it bought. Nothing
outside `rakshak.eval` constructs them; if that stops being true, they move.

### `07-feature-register.md`'s own counts are internally inconsistent
The document states T1 = 23 and T2 = 9. Its own tables list 28 and 8. The "+14 residuals"
figure is likewise stale — 21 features now carry `has_cohort_residual`. The register was not
corrected during the sprint because it is an input document and editing it mid-flight would
have invalidated the diff against the implementation that is its purpose.

---

## 5. Findings about the tests themselves

**Parity is necessary and nowhere near sufficient.** T-120 found a real bug that the parity
suite was structurally incapable of catching: the warmup window was anchored on
`onboarded_at`, but the event stream starts at day 0 while merchants onboarded up to 120 days
earlier — so for most of the population the warmup window contained no events at all, and
every baseline was empty. Both runners agreed perfectly on the wrong answer, and parity
stayed green throughout. It was found only when real generator output arrived.

**Parity says two runners agree. It never says they agree about something meaningful.** The
fix re-anchors warmup to the first *observed* event date. The lesson belongs in the writeup:
the dual-runner design protects against one specific failure — a feature that cannot be
served — and buys nothing at all against a feature that is simply wrong.
