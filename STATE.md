# STATE — Rakshak

PHASE:        4 → 5 — TICKET complete, EXECUTE not started
LAST SESSION: 2026-08-28 — Phases 0–4 compressed into one pass. Grill and lit-survey were already complete; transcribed rather than re-run.
NEXT ACTION:  Execute **T-0001** (repo scaffold + determinism guard). Then T-0002, which retires the highest risk in the project.

## Load for next session
- `CLAUDE.md`
- `11-tickets/BOARD.md`
- `11-tickets/T-0001.md`

Nothing else. T-0001 names its own context.

## Countdown
| Date | Milestone |
|---|---|
| Sat 29 Aug | T-0001 → T-0004. **HMM must recover known states by end of Saturday.** |
| Sun 30 Aug | T-0005 → T-0008. Baselines + eval harness + cost layer. |
| Mon 31 Aug – Mon 1 Sep | T-0009 → T-0014. Frontier, ablations, BAF validation, README. |
| Mon 1 Sep EOD | **Code freeze.** `make eval` green. README numbers final. |
| Tue 2 – Wed 3 Sep | Video. |
| Thu 4 Sep | Review. |
| Fri 5 Sep | Submit. |

If Monday arrives at 60%, cut in this order (last cut first): dashboard → T-0010 BOCPD → T-0009 NSGA → T-0008 shrinkage → T-0007 cost layer. **T-0001 through T-0006 are not cuttable** — they are the spine of the video.

## Settled — do not relitigate

| Decision | Where | Reversibility |
|---|---|---|
| Track 02, AI Risk Manager | 00-charter.md | one-way door — chosen |
| Post-onboarding merchant drift, not transaction-level fraud | 00-charter.md | one-way door — this is the whole differentiator vs Vulcan |
| Hand-written HMM, not `hmmlearn` | ADR-0001 | cheap |
| GNN and transformers rejected | ADR-0002 | costly to reverse in time available |
| RL rejected for the build, kept as a pitch slide | ADR-0003 | cheap |
| NSGA-II, not NSGA-III | ADR-0004 | cheap |
| Three actions: pass / review / hold | ADR-0005 | costly |
| Explicit analyst-hour capacity constraint | ADR-0005 | costly |
| Per-merchant thresholds via empirical-Bayes shrinkage | ADR-0006 | cheap |
| Four typologies injected: bust-out, laundering endpoint, category drift, refund collusion | 06-requirements.md §FR-004 | costly |
| Slow-ramp evader is an adversarial test we expect to partially fail | 06-requirements.md §FR-005 | cheap |
| Hybrid data: own generator (sequence layer) + BAF (decision layer) | ADR-0007 | cheap |
| Solo build | 00-charter.md | fixed |

## Open branches
- **Rakshak legacy code** — prior work under this name was not recoverable at planning time. If it surfaces, salvage the eval harness only; the transaction-level framing is superseded. Deferred to T-0001.
- **BAF download size** — Base variant is ~1M rows. If it will not load in reasonable time, subsample with a fixed seed and document it. Deferred to T-0012.
- **K (number of HMM states)** — provisionally 4. Settled empirically by BIC sweep in T-0004.
- **Cost matrix values** — provisional figures in `07-math.md §5` with sources. Sensitivity analysis in T-0011 is what makes them defensible.

## Highest unretired risk

**A-002 — that a hand-written HMM will recover injected latent states on our own generated data at all.**

If T-0004 cannot recover known states on data where we control the ground-truth state path, the entire sequence layer is dead and the project must DESCEND to Phase 2 and fall back to LightGBM-on-windowed-aggregates plus the cost layer — still a viable, honest submission, but it loses the explainability differentiator. **T-0004 must complete on Saturday.** Do not let it slip.

Second-highest: **A-005** — that the HMM beats the LightGBM baseline. It may not. That outcome is survivable and pre-scripted (see `10-done.md §Verdict`), but only if the ablation table exists, which is why T-0005 and T-0006 come before any sophistication.

## Deferred questions raised out of phase
- "Should the dashboard be Streamlit or static matplotlib?" — Phase 5, T-0014. Default is static; a broken Streamlit app is worse than a clean PNG.
- "Can we call Razorpay test-mode APIs for realism?" — not required by Track 02. Only revisit if T-0001..T-0012 finish early.
