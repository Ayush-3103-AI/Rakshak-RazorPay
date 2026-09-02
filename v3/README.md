# Rakshak v2 — post-onboarding merchant risk sentinel

Razorpay scores every *transaction* (Vulcan) and reviews every *merchant* once, at
onboarding (Bumblebee). Nothing watches a merchant that already cleared onboarding as its
behaviour drifts over the following weeks. Rakshak is that sentinel: every day, for every
cleared merchant, it emits `PASS` / `REVIEW` / `HOLD` under a hard analyst-capacity budget,
with a merchant-readable reason attached to every non-`PASS`.

This tree (`ver-2/v3`) is v2 of the project — see `CLAUDE.md` for the full architecture,
stack, and the prime directives (frozen eval harness, immutable v1 results, no ground-truth
leakage into features/models). Details in `LIMITATIONS.md` (honest failures, with numbers)
and `project-context/STATE.md` (current resume point).

## Cycle 4

Cycle 4 regenerated the dataset at full scale (40,000 merchants × 365 days, 121.9M
transactions), rescored the entire 80-row ladder across five seeds instead of one, built
and scored Rungs 5–7 (previously deferred, reversed by GitHub #51 — see `LIMITATIONS.md`
§9.9), and ran the pre-registered gate that decides whether the test split opens.

**The test split stayed shut.** The floor-fail gate failed on both its conjuncts — 0/5
seeds and 0/5 cost-asymmetry ratios (`EVAL-LOCK-CYCLE4.json`, `open_count: 0`) — because it
was anchored to a stale cycle-3 threshold (0.7017) invalidated by cycle 4's own regeneration
(the real floor is 0.5240): an acknowledged pre-registration error, not a re-anchor after
the fact. Every number in `docs/results_v2.md` is validation-split only.

Headline findings, all with numbers in `LIMITATIONS.md` §9:
- Correcting exposure lifts savings on 5 of 5 rungs (§9.2); cycle 3's "Rung 4 cut" was an
  artefact of the exposure defect it fixed, and Rung 4 is now the best savings rung (§9.7).
- Rung 5 has the best PR-AUC on the ladder but near-worst savings — a calibration problem,
  not a ranking one (§9.9).
- `make all`'s `parity` and `perf` stages had been silently red for a cycle and a half
  before this cycle, and the one remaining red test (`test_cohort.py`, a 14.3%-vs-15%
  threshold drift) is now resolved — the tree is fully green and K-5 is retired in fact,
  not just recorded as retired (§9.10).
- **The cost-asymmetry sweep, run for the first time (§10).** `sweep_cost_asymmetry` had
  been in the tree, unit-tested, since T-132 and had never been run over the ladder — so
  every savings number the project has published was a single point estimate at one cost
  matrix, and half of a pre-registered gate had no input to read. Run now
  (`docs/results/cost_sweep.md`): Rung 4 holds **+0.5853 to +0.6001 across four orders of
  magnitude** of false-hold/fraud-loss asymmetry and beats the `volume_rank` floor at 5 of
  5 ratios, with the shipped cost matrix inside the swept grid. The gate's verdict does not
  move; how completely it was evaluated does.
- **Where that margin comes from, decomposed (§10.3), because it is not the ranking.** With
  HOLD made unreachable and nothing else changed, Rung 4's margin over the floor falls from
  +0.0740 to +0.0403 — it still wins, at 5/5 ratios, but the pricing asymmetry the
  pre-registration disclosed (§4.3) is worth about 45% of it. Priced as a raw REVIEW-only
  ranking, **every rung loses to `volume_rank`**, and the best pure rupee-ranker among them
  is Rung 1, the rule engine. The advantage is a decision-layer result, not a modelling one.

## Reproducing

```bash
uv sync
make all      # lint → parity → gen → gates → perf → test; must pass from a clean clone
make report   # regenerate docs/results_v2.md from the frozen eval
```

`make eval` refuses to run against the locked test split unless `RAKSHAK_UNLOCK=1` is set.
It is not set anywhere in this repo.
