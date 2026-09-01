# Cycle 4 — the metric could not fire, and the floor was winning on exposure

**Status:** steps 1–5 of 8 done; ladder rescore in flight · **Session:** 2026-09-01/02
**Loaded:** `STATE.md`, `12-spec-cycle4.md`, `LIMITATIONS.md` §8, `eval/capacity.py`,
`generator/labels.py`, `generator/config.py`, `models/rung0_floors.py`,
`features/tier1.py::DeclaredRatio`, `configs/rung_roster.yaml`, surveys `13a`/`13b`/`13c`

## Built

- `configs/scenario_v2.yaml` — onsets 30–240 → **30–364**, 20,000 → **40,000** merchants,
  per-typology bounds affine-rescaled, `label_resolution_horizon_day: 500`.
- `generator/labels.py` — `emit_labels(..., label_resolution_ns=None)`, defaulting to
  `sim_end_ns`. One line. Backward compatibility asserted, not claimed.
- `tests/unit/test_config.py` — `test_every_evaluated_split_contains_in_window_drift_onsets`
  and three siblings.
- `LIMITATIONS.md` §8.7a and §8.3a — the two findings, with their arithmetic.
- `docs/adr/ADR-V3-001-no-autograd.md` — cited in four places, existing as a file for the
  first time.
- `docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`, `EVAL-LOCK-CYCLE4.json` (`open_count: 0`).
- `models/decision_realised_exposure.py`, `models/rung9_rank_cusum.py`,
  `scripts/rescore_cycle4.py`, `scripts/cycle4_verdict.py`.

## Surprised

**1. The headline metric could not fire, and seven policies scoring identically was the
tell nobody read.** `detection_rate_d7` was 0.000 for four rungs and three floors, reported
as "nothing detects anything quickly." It was 0.000 because drift onsets stopped at day 217
and the window opened on day 240: **0 of 294 fraud merchants could have scored d7**, and an
oracle scores 0.000 too. The surprise is not the defect — it is that *seven different
policies producing the same number to three decimals* did not read as suspicious at the
time. A metric that cannot discriminate looks exactly like a metric on which nothing
discriminates.

**2. The existing geometry guard passed, correctly, and was the wrong guard.**
`test_test_split_has_enough_labelled_positives_per_seed` counts *labelled positives in a
split*. A merchant can carry a resolved positive label in a split without ever having
onsetted inside it. One noun apart from the property that mattered. The new guard counts
in-window onsets and goes red on the cycle-3 config for all five seeds — checked in both
directions, because a guard that has only ever been seen to pass is decoration.

**3. Three literature surveys, and none of them found the second failure.** Quickest change
detection, delayed-feedback/PU learning, and the cost/decision layer all ran in parallel and
all produced good work. The finding came from reading `capacity.py` to answer a question one
survey *raised* — does the decision layer rank by probability or by expected value? It
already ranked by expected value, `0.8·p·exposure − 250`, which meant the textbook fix was
already in place and the difference had exactly one remaining place to live: the exposure
term. Survey 13c reached the same mechanism independently by tracing source, which is what
turned one reading of the code into a finding two methods agree on.

**The measurement, once the hypothesis was named, took four minutes.** Against realised
loss over 294 fraud merchants: `declared_monthly_gmv` ρ = **0.533**, observed GMV ρ =
**0.935**. At K = 15, exposure *alone* captures 20.5% of total fraud loss when declared and
**42.7%** when observed, against an oracle's 46.2%. `volume_rank` was never a dumb floor
beating clever models — it is a ρ = 0.94 exposure estimator beating models whose excellent
`p` was multiplied by a ρ = 0.53 one.

**And the first version of those numbers was wrong, which is why the script exists.**
Writing `scripts/exposure_diagnostic.py` so a reader could regenerate them caught that the
original measurement filtered `status == "captured"` but did not exclude refunds, while
`cli._observed_volume` — the quantity `volume_rank` is actually handed — excludes both. The
observed column was understated (ρ 0.929, 37.83% at K = 15). Both corrections make the gap
*wider*. The lesson is not "check your filters": it is that a number nobody can regenerate
is a number nobody can correct, and this one was in a graded artefact for three commits.

**4. The fix needed no new feature.** `v_declared_ratio` is defined as
`trailing-30d GMV ÷ declared_monthly_gmv`, so `v_declared_ratio × declared_monthly_gmv` *is*
trailing-30d realised GMV. Both factors were already registered, already point-in-time,
already past the leakage gate. The correction is a `dataclasses.replace` on one field.

**5. Rung 9 accumulates on level, not on change — and its own diagnostic said so
immediately.** A smoke run put **15.4% of merchant-days at the accumulator cap** and
`alert_jaccard_wow` at **1.000** — the exact static-watchlist degeneracy the cap exists to
prevent. The reason is structural rather than a bug: the Page recursion assumes mean-zero
increments under the null, but a merchant's *cross-sectional rank* is persistent. A
consistently high-risk merchant sits near the top of its cohort every day, contributes a
positive normal score every day, and the accumulator ramps without anything having changed.
The specified method ranks the incumbent *score*; what a change detector needs is a
quantity that is mean-zero when **that merchant** is stable. Recorded, run as
pre-registered, and **not tuned** — the pre-registration says a rung that fails its gate is
reported as failing. `accumulator_frac_at_cap` is written onto every result row so the
claim is checkable rather than narrated.

**6. The realised TTD denominator came in at the bottom of its predicted range.** The
pre-registered Monte Carlo said 7–14 in-window resolved onsets in the validation fold; the
generated data gives **7**, for a standard error near **19 pp** rather than the ~13 pp
declared. Inside the range, so the pre-registration stands — but the latency comparison is
even less powered than it was declared to be, and every TTD number must carry that
denominator.

**7. The rescore's binding constraint is a dtype, not the models.**
`dataset.load_panel` ends with `.to_numpy().astype(np.float64)`, so every process that
touches the panel materialises **6,146,940 × 49 × 8 bytes = 2.24 GiB** before it selects a
split or a column subset. Three concurrent jobs is therefore ~7 GiB of panel alone on a
16 GB machine, and a fourth process attempting the same load raised
`numpy._core._exceptions._ArrayMemoryError` outright. The cube this panel was *written*
from is `float32` (`dataset.build_panel`: "float32 halves the peak and is what LightGBM bins
to anyway"), so the widening on read is pure cost — it doubles the footprint to reach a
precision the writer deliberately declined and the consumer discards at binning time.

Not changed during cycle 4, and the reason is not caution about the edit. Narrowing the read
to `float32` would shift the low-order bits of every feature, so models trained before the
change and models trained after it would not be comparable — and a ladder with
mixed-provenance rows is the thing this cycle exists to avoid. It is a one-line change for a
cycle that begins with a clean rescore, and it belongs in the next one.

## Broke

- `n_merchants` and `onset_window_max_day` are asserted as constants in
  `test_charter_section_10_parameters`. Cycle 4 moves both, so the guard was updated — with
  the old values written into its docstring, because an amendment nobody can see is a
  silent edit.
- The first `git stash` diagnosis of a red cohort test was worth doing: **the failure
  pre-dated cycle 4 with byte-identical numbers.** Without checking, an unrelated red test
  would have been attributed to the config change and "fixed" by weakening its threshold.
- `rung8_realised_exposure.py` collided with `tpp_hawkes_nb`, which already owns Rung 8 in
  the roster. Renamed to `decision_realised_exposure.py` — and it should never have had a
  number, since the pre-registration registers it as an A/B over the whole ladder rather
  than as a competing rung. Recorded as a dated clerical note at the foot of the sealed
  document instead of applied silently.
- Twice, `cd` inside a compound shell command left the working directory in `ver-2` rather
  than `ver-2/v3`, and one `uv run` built the wrong package into a stray venv. Harmless
  both times, caught by checking `rakshak.__file__`. Worth knowing that the two trees are
  close enough to confuse a command but far enough apart to produce nonsense.

## Decisions worth reviewing

**The eval package was not edited, and that is the deliverable.** `eval_module_sha256` is
`c009e38d…` in both the cycle-3 and cycle-4 locks. The same hash-verifiable harness that
scored the failing ladder scores the new one, so "we only moved the data" is checkable
rather than asserted. `time_to_detection` and `detection_rates` were never the defect.

**The exposure correction is an A/B, not a silent fix.** Arm A keeps cycle 3's wiring
exactly — asserted by a byte-identity test — so the geometry effect and the wiring effect
are separated rather than confounded. A wiring fix folded quietly into a regeneration would
have made the cycle-3 → cycle-4 comparison unreadable.

**Two spec defects were raised rather than patched.** The spec's predicted test-fold yield
was wrong by ~40%, and the affine rescale it mandates leaves **R2 and R9 — 25% of the fraud
mix, including the slow-ramp bust-out v1 failed on — with zero probability of onsetting in
either evaluation window.** Widening those windows would make two typologies easier relative
to their peers, which the spec explicitly rejected. The decision stands; its cost is
recorded in the pre-registration and must be reported as *structurally absent*, never as
zero.

**A comparison asymmetry was disclosed and deliberately left broken.** Floors are priced
REVIEW-only at ₹250 per error; rungs are priced on their own actions, which may HOLD at
₹8,250 — a 33× difference. `savings_of_ranking`'s docstring claims the comparison differs
only in the score vector, which is true floor-vs-floor and false floor-vs-rung. Fixing it
means editing the locked eval package. Named, not fixed, hash left alone.
