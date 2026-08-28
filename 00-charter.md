<!-- HEAD
FILE:     00-charter.md
PHASE:    0 — CHARTER
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  Rakshak detects post-onboarding merchant risk drift and decides pass/review/hold
          under an analyst-hour budget. Built for Razorpay AI Buildathon Track 02, solo,
          CPU-only, 4-day build window, submission 5 Sep 2026. Success = beat a static rule
          engine on savings score at equal review budget, on a leakage-free held-out split,
          with a reported failure mode. Kill criteria: HMM cannot recover known states by
          Sat EOD, or no baseline is beaten by Mon EOD. Non-goals: transaction-level fraud
          scoring, onboarding-gate review, RTO/COD, KYC deepfake detection, any GPU method.
OPEN:     none
-->

# 00 — Charter

## 1. Problem, one sentence

Razorpay scores every transaction in real time (Vulcan) and reviews every merchant once at onboarding (Bumblebee), but nothing watches a **cleared** merchant's behaviour drift from good to bad in the weeks that follow — so bust-outs, laundering endpoints, category drift and refund collusion stay invisible until chargebacks land 45–120 days later, while blunt global thresholds simultaneously freeze honest merchants at a cost widely estimated at ₹400–600 for every ₹100 of fraud prevented.

**Who hurts:** (a) Razorpay's risk P&L, which absorbs the loss; (b) honest merchants whose settlements are frozen — the single most common complaint in Razorpay's public reviews; (c) risk operations analysts, who Razorpay's own engineering blog reports were spending 700–800 hours a month on manual review.

## 2. Success metric with a number

> **Rakshak beats a static velocity/refund-ratio rule engine by ≥20% relative on the Bahnsen savings score, at equal analyst-hour budget, on a temporally-and-group-split held-out set of unseen merchants.**

Falsifiable. Could come out false. Measured by `make eval`, seed-fixed, reproducible by a stranger who clones the repo.

Secondary, reported but not gating:
- **Detection lag** — median days between injected transition and first flag, versus the rule engine.
- **Gap-to-oracle** — savings achieved as a fraction of the perfect-foresight knapsack ceiling.
- **Explainability coverage** — % of flagged merchants for which a human-readable reason string is produced from the Viterbi path.

## 3. Kill criteria

| Trigger | Observation | Response |
|---|---|---|
| **K1** | HMM cannot recover injected states on generator data with known ground-truth state paths, by Sat 29 Aug EOD | DESCEND to Phase 2. Drop the sequence layer. Ship LightGBM + cost layer + capacity constraint. Still a valid Track 02 submission. |
| **K2** | Rakshak does not beat the static rule engine on savings by Mon 1 Sep EOD | Do **not** tune to win. Report the negative result, pivot the narrative to explainability and the cost frontier, and say so on camera. |
| **K3** | `make eval` cannot run end-to-end in under 15 min on the build laptop | Cut typologies from 4 to 2 and subsample merchants. Reproducibility beats scale. |
| **K4** | Submission deadline moves earlier than 5 Sep | Freeze at whatever ticket is complete. A working T-0006 with honest numbers beats a broken T-0012. |

## 4. Hard constraints

- **Deadline:** submission 5 Sep 2026 (verify on form). Code freeze Mon 1 Sep EOD.
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
- A production-grade UI. A clean matplotlib figure outranks a half-broken web app.
- Real Razorpay API integration. Not required by Track 02.

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
