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

**The test split stayed shut.** The floor-fail gate failed 0/5 seeds
(`EVAL-LOCK-CYCLE4.json`, `open_count: 0`) because it was anchored to a stale cycle-3
threshold (0.7017) invalidated by cycle 4's own regeneration (the real floor is 0.5240) —
an acknowledged pre-registration error, not a re-anchor after the fact. Every number in
`docs/results_v2.md` is validation-split only.

Headline findings, all with numbers in `LIMITATIONS.md` §9:
- Correcting exposure lifts savings on 5 of 5 rungs (§9.2); cycle 3's "Rung 4 cut" was an
  artefact of the exposure defect it fixed, and Rung 4 is now the best savings rung (§9.7).
- Rung 5 has the best PR-AUC on the ladder but near-worst savings — a calibration problem,
  not a ranking one (§9.9).
- `make all`'s `parity` and `perf` stages had been silently red for a cycle and a half
  before this cycle, and the one remaining red test (`test_cohort.py`, a 14.3%-vs-15%
  threshold drift) is now resolved — the tree is fully green and K-5 is retired in fact,
  not just recorded as retired (§9.10).

## Reproducing

```bash
uv sync
make all      # lint → parity → gen → gates → perf → test; must pass from a clean clone
make report   # regenerate docs/results_v2.md from the frozen eval
```

`make eval` refuses to run against the locked test split unless `RAKSHAK_UNLOCK=1` is set.
It is not set anywhere in this repo.
