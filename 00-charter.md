<!-- HEAD
FILE:     00-charter.md
PHASE:    0 — CHARTER
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Rakshak detects post-onboarding merchant risk drift and decides pass/review/hold
          under an analyst-hour budget. Built for Razorpay AI Buildathon Track 02, solo,
          CPU-only, 4-day build window, submission 5 Sep 2026. Success = beat a static rule
          engine on savings score at equal review budget at the cited central cost asymmetry,
          on a leakage-free held-out split, with the failure boundary and a reported failure
          mode. Sections 2 and 7 carry dated T-0017 amendments (2026-08-28). Kill criteria:
          HMM cannot recover known states by Sat 29 Aug EOD, or no baseline is beaten by
          Tue 1 Sep EOD. Non-goals: transaction-level fraud
          scoring, onboarding-gate review, RTO/COD, KYC deepfake detection, any GPU method.
OPEN:     none
-->

# 00 — Charter

## 1. Problem, one sentence

Razorpay scores every transaction in real time (Vulcan) and reviews every merchant once at onboarding (Bumblebee), but nothing watches a **cleared** merchant's behaviour drift from good to bad in the weeks that follow — so bust-outs, laundering endpoints, category drift and refund collusion stay invisible until chargebacks land 45–120 days later, while blunt global thresholds simultaneously freeze honest merchants at a cost widely estimated at ₹400–600 for every ₹100 of fraud prevented.

**Who hurts:** (a) Razorpay's risk P&L, which absorbs the loss; (b) honest merchants whose settlements are frozen — the single most common complaint in Razorpay's public reviews; (c) risk operations analysts, who Razorpay's own engineering blog reports were spending 700–800 hours a month on manual review.

## 2. Success metric with a number

> **Rakshak beats a static velocity/refund-ratio rule engine by ≥20% relative on the Bahnsen
> savings score at the cited central cost asymmetry, at equal analyst-hour budget, on a
> temporally-and-group-split held-out set of unseen merchants — with the relative improvement
> reported across the full plausible asymmetry range and the boundary at which the claim fails
> stated explicitly.**

Falsifiable. Could come out false. Measured by `make eval`, seed-fixed, reproducible by a stranger who clones the repo.

> ### AMENDMENT — §2 made cost-conditional · dated 2026-08-28 · ticket T-0017
>
> **What it replaced.** The sentence previously read, in full: *"Rakshak beats a static
> velocity/refund-ratio rule engine by ≥20% relative on the Bahnsen savings score, at equal
> analyst-hour budget, on a temporally-and-group-split held-out set of unseen merchants."*
> Unconditional — no mention of the cost asymmetry the savings score is computed under.
>
> **Why, and why now.** The Bahnsen savings score is a function of the cost matrix
> (`07-math.md` §5), and the cost matrix's primitives are sourced estimates with real
> uncertainty. The ≥20% margin is therefore a function of the false-positive-to-fraud-loss
> asymmetry, and it may hold at one end of the plausible range and fail at the other. T-0007b
> sweeps exactly that; T-0011 renders K2's verdict.
>
> **This amendment was made on 2026-08-28, BEFORE T-0007b ran and before any swept number
> existed.** That ordering is the whole point. Amending §2 after seeing "wins above X, loses
> below X" would read as an excuse no matter how honest the intent; amending it first is
> pre-registration. Same discipline as the K1 story, where the RAMP-recall ≥ 0.35 bar was
> recorded before measuring, failed at 0.234, and was committed as a strict `xfail` rather
> than quietly dropped.
>
> **The ≥20% threshold itself is unchanged.** Only its conditionality on the cost asymmetry is
> being made explicit in advance. Softening the bar is not permitted by this amendment and was
> not done.

Secondary, reported but not gating:
- **Detection lag** — median days between injected transition and first flag, versus the rule engine.
- **Gap-to-oracle** — savings achieved as a fraction of the perfect-foresight knapsack ceiling.
- **Explainability coverage** — % of flagged merchants for which a human-readable reason string is produced from the Viterbi path.

## 3. Kill criteria

| Trigger | Observation | Response |
|---|---|---|
| **K1** | HMM cannot recover injected states on generator data with known ground-truth state paths, by Sat 29 Aug EOD | DESCEND to Phase 2. Drop the sequence layer. Ship LightGBM + cost layer + capacity constraint. Still a valid Track 02 submission. |
| **K2** | Rakshak does not beat the static rule engine on savings by Tue 1 Sep EOD | Do **not** tune to win. Report the negative result, pivot the narrative to explainability and the cost frontier, and say so on camera. |
| **K3** | `make eval` cannot run end-to-end in under 15 min on the build laptop | Cut typologies from 4 to 2 and subsample merchants. Reproducibility beats scale. |
| **K4** | Submission deadline moves earlier than 5 Sep | Freeze at whatever ticket is complete. A working T-0006 with honest numbers beats a broken T-0012. |

## 4. Hard constraints

- **Deadline:** submission 5 Sep 2026 (verify on form). Code freeze Tue 1 Sep EOD (1 Sep 2026 is a Tuesday).
- **Compute:** CPU only. No GPU of any kind.
- **Team:** solo.
- **Data:** no real merchant data. Public datasets + own generator only.
- **Regulatory posture:** defense-only. Anything offense-capable is disqualifying per the track rules.
- **Licence:** MIT / BSD / Apache-2.0 dependencies only.
- **Runtime:** `make eval` < 15 minutes.
- **Output:** public repo + 5-minute video + architecture doc. All three are graded.

## 5. Project type

**Hybrid — research-flavoured software with a hiring-submission wrapper.** This matters: the artifact is judged by a panel, not by users, so measurement honesty and explainability carry as much weight as raw performance. Phase 2 spec format follows the research pattern (baselines, oracle, frozen eval) rather than the product pattern.

## 6. Doctrine hook

No organisational doctrine skill applies. This is a personal submission, not an Ayudh Energy project. The framework's own rules govern.

**But one external doctrine is binding: the Razorpay Track 02 published bar.** Quoted here so it is never paraphrased from memory:

> *"Build a working detector, verifier or auto-responder for one class of loss, with measured precision and recall on a held-out test set. The bar: Honest metrics including false-positive cost. Strictly defense-only: anything offense-capable is disqualified."*

Every requirement in `06-requirements.md` traces to a clause in that paragraph.

## 7. Non-goals

Explicitly out of scope. Reopening any of these is a DESCEND, not a quick addition.

- Transaction-level fraud scoring — **Vulcan owns this.**
- Onboarding-gate merchant review — **Bumblebee owns this.**
- RTO / COD return fraud — **Thirdwatch owns this**, and it is the most crowded idea in the submission pool.
- KYC deepfake / synthetic document detection — needs vision models and a GPU.
- Chargeback evidence generation — demos well, measures badly; no ground truth for "would this have won."
- Any method requiring a GPU.
- A production UI **inside the build window**. A read-only results viewer, built after code
  freeze and rendering only committed artifacts from `results/`, is permitted as a presentation
  asset.
- Real Razorpay API integration. Not required by Track 02.

> ### AMENDMENT — §7 UI non-goal narrowed, not reversed · dated 2026-08-28 · ticket T-0017
>
> **What it replaced.** The bullet previously read, in full: *"A production-grade UI. A clean
> matplotlib figure outranks a half-broken web app."* That wording is **not** deleted and its
> reasoning still stands — it was about a half-built web app competing with measurement work
> for the four build days. After code freeze it cannot compete with anything, because there is
> nothing left to measure.
>
> **Why now.** The repo contradicted itself. §7 forbade a production UI while `T-0014` built
> one on explicit instruction and said so in its own text. A panel reader hits both. The
> contradiction is resolved by narrowing the non-goal to the build window rather than by
> reversing it — reversing it would discard a correct piece of reasoning to fit a schedule.
>
> **Binding conditions on the permitted viewer**, carried into T-0014: built 2–3 Sep, in the
> video window, outside the build window; read-only; renders committed artifacts from
> `results/` and computes nothing. A viewer that calculates is a second implementation that
> can disagree with the README.

## 8. Provisional cast

| Role | Status |
|---|---|
| Grill / plan interrogation | `[have]` — `grill-me`, already run |
| Literature survey | `[have]` — `lit-survey`, already run |
| Project decomposition | `[have]` — `elite-project-manager` |
| Test / eval engineering | `[have]` — `ayudh-test-engineer` (transferable: pytest architecture, golden files, CI gates, eval harness discipline) |
| Video / narrative craft | `[have]` — `arka-pitch-deck` for structure, `ayudh-comms-chief` for wording |
| Quant sequence modelling | `[have]` — `quant-alpha-research` (HMM regime detection), `quant-portfolio-risk` (cost/capacity framing) |

**Cast gaps:** none that pass the two-of-three test. A "payments-risk domain expert" skill would be invoked more than three times, but the domain knowledge is already captured in `03-landscape.md` and would not be reused on a second project. Handled inline. Recorded here so it is not rebuilt.
