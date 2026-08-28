<!-- HEAD
FILE:     01-understanding.md
PHASE:    1a — UNDERSTAND (grill output)
UPDATED:  2026-08-28
STATUS:   gated
SUMMARY:  17 settled decisions from the grill, each with reversibility. Key one-way doors:
          Track 02; merchant-level not transaction-level; three-action policy under a
          capacity constraint. Key rejections: RL (no reward signal), GNN (GPU + circular
          eval), transformers (vendor research reports parity). 9 assumptions tagged;
          A-002 (HMM recovers injected states) is the project-killer and T-0004 attacks it
          first. Open branches: K state count, cost matrix values, BAF load feasibility.
OPEN:     A-002, A-005 untested. K and cost values settled empirically in Phase 5.
-->

# 01 — Understanding

## Settled decisions

| # | Decision | Chosen | Rationale | Reversibility |
|---|---|---|---|---|
| D-01 | Track | 02 — AI Risk Manager | Cleanest path to a defensible held-out number; the published bar rewards measurement discipline, which is our strength | one-way door |
| D-02 | Unit of analysis | Merchant, over weeks | Vulcan owns transaction-level; merchant-over-time is the documented gap in Razorpay's own engineering blog | one-way door |
| D-03 | Relationship to Vulcan | Consume, don't compete | Vulcan's per-transaction risk score becomes an input feature. Architecture diagram must show this. | costly |
| D-04 | Sequence model | Hand-written HMM | Latent-state structure matches the problem; Viterbi path is a native audit trail | costly |
| D-05 | Changepoint alternative | BOCPD as measured baseline, not primary | HMM's named states are the business differentiator; BOCPD gives only "something changed" | cheap |
| D-06 | RL | Rejected for build, retained as pitch slide | No reward signal — ground truth lags 45–120 days; training on our own generator learns our assumptions | cheap |
| D-07 | GNN | Rejected | GPU + circular synthetic-graph evaluation. Approximated with graph-derived scalar features. | costly |
| D-08 | Transformer | Rejected | NICE Actimize (arXiv:2605.21490) reports parity with feature engineering | cheap |
| D-09 | Action space | Three: pass / review / hold | Mirrors Razorpay's actual operations; makes "review" a distinct cost (analyst time) | costly |
| D-10 | Capacity | Explicit constraint, review-hours ≤ K | Converts threshold-picking into constrained policy optimisation; this is what makes the project distinctive | costly |
| D-11 | Calibration | Per-merchant via empirical-Bayes shrinkage toward segment prior | Correct answer to the cold-start objection; new merchants inherit segment economics | cheap |
| D-12 | Optimiser | pymoo NSGA-II | 3 objectives is not many-objective. GA justified only because capacity couples merchants — prove it against a grid-search baseline. | cheap |
| D-13 | Data strategy | Hybrid — own generator for sequences, BAF for decision layer | Converts "I made it up" into "sequence layer synthetic, decision layer on a peer-reviewed public benchmark" | cheap |
| D-14 | Typologies | Bust-out, laundering endpoint, category drift, refund collusion | Four is enough for credible per-class metrics; five spreads thin | costly |
| D-15 | Adversarial test | Slow-ramp evader, expected to partially fail | Reporting a measured failure is the strongest honesty signal available | cheap |
| D-16 | Eval protocol | Temporal split + merchant-group split, both enforced in code | Random splits are the standard self-deception in this domain | one-way door |
| D-17 | Scope cut order | dashboard → BOCPD → NSGA → shrinkage → cost layer | Decided while calm, before the Monday panic | cheap |

## Open branches

| Branch | Blocked on | Settled by |
|---|---|---|
| K (HMM state count) | Empirical BIC sweep over K ∈ {2..6} | T-0004 |
| Cost matrix values | Sensitivity analysis; provisional values in `07-math.md §5` | T-0011 |
| BAF subsampling | Whether ~1M rows load in acceptable time | T-0012 |
| Legacy Rakshak code salvage | Whether the prior repo surfaces | T-0001 |
| Dashboard vs. static figures | Time remaining Monday | T-0014 |

## Assumptions

Tagged. The `[assumed]` list is what the ticket order attacks.

| # | Assumption | Tag | Attacked by |
|---|---|---|---|
| **A-001** | A synthetic generator can produce merchant streams whose statistics are plausible enough that findings transfer | `[assumed]` | T-0002 — plausibility checks against published aggregate stats |
| **A-002** | A hand-written HMM will recover injected latent states on data where we control the ground-truth path | `[assumed]` | **T-0004 — highest-risk ticket, must land Saturday** |
| **A-003** | Within-merchant standardisation of emissions is what prevents the 2008-era false-positive failure | `[assumed]` | T-0003, validated in T-0006 |
| **A-004** | Graph-derived scalar features capture enough of the relational signal to detect laundering endpoints without a GNN | `[assumed]` | T-0006 ablation: graph features on/off |
| **A-005** | The HMM beats LightGBM-on-windowed-aggregates | `[assumed]` | T-0005 + T-0006. **May be false. Outcome is pre-scripted.** |
| **A-006** | The capacity constraint genuinely couples per-merchant thresholds enough to justify a GA | `[assumed]` | T-0009 — must beat the uncoupled grid-search baseline or the GA is decoration |
| **A-007** | Empirical-Bayes shrinkage improves over a global threshold at low merchant volume | `[assumed]` | T-0008 ablation |
| **A-008** | Razorpay's published risk-ops figures (10–12k reviews/month, ~4 min each) are a fair basis for the capacity constant | `[verified]` — Razorpay Engineering, Dec 2025 | — |
| **A-009** | The submission deadline is 5 Sep 2026 | `[unknown]` — one third-party listing only, not on the official page | **Verify on the form today.** |

## Design-tree branches deliberately not explored

Recorded so a future session does not reopen them thinking they were missed.

- Federated / privacy-preserving learning across merchants — interesting, irrelevant at this scale, no time.
- Conformal prediction for calibrated abstention — genuinely well-matched to the three-action design and worth a sentence in the video as future work, but adds a day.
- Multi-expert routing (which analyst gets which case) — the FiFAR dataset exists for exactly this. Out of scope; one analyst pool assumed.
- Active learning from analyst overrides — this is the feedback loop whose absence killed the RL option. Mention as the thing Razorpay's data would unlock.
