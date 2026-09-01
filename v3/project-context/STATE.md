<!-- HEAD
FILE:     STATE.md
PHASE:    resume point
UPDATED:  2026-09-02
STATUS:   live — update at the end of EVERY session
SUMMARY:  The resume point for Rakshak v3. Read this first, every session, before anything
          else. It names the current cycle, what is running, and the exact next action.
          Nothing else should be loaded speculatively.
OPEN:     The cycle-4 ladder rescore is IN FLIGHT. Three findings are banked and committed
          and none of them depends on it landing.
-->

# STATE — Rakshak, cycle 4

**Cycle:** 4 · **Phase:** 5 (execute) · **Freeze:** 3 Sep 2026 · **Submission:** 5 Sep

---

## Where things stand

**Cycle 3 is complete and tagged `cycle3-ladder-immutable`.** Its numbers are frozen.
Cycle 4 is regenerating the dataset underneath the same harness.

**Cycle-4 protocol** (`project-context/12-spec-cycle4.md`), steps 1–5 of 8 done:

| # | step | status |
|---|---|---|
| 1 | tag the cycle-3 ladder immutable | ✅ `cycle3-ladder-immutable` |
| 2 | write the survey, no code during it | ✅ three surveys, `13a` / `13b` / `13c`, ran in parallel |
| 3 | pre-register cycle 4 | ✅ `docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`, committed `2ee6972` |
| 4 | seal the lock, recording the previous as superseded | ✅ `EVAL-LOCK-CYCLE4.json`, `open_count: 0` |
| 5 | regenerate the dataset | ✅ 40,000 × 365, 121.9M txns, 2.7 GB; panel 6.15M rows |
| 6 | rescore the full ladder, every floor and every rung | ⏳ **IN FLIGHT** — 80 jobs, both arms, 5 seeds |
| 7 | implement and score the new rung | ✅ built + wired; Rung 9 is cuttable |
| 8 | open the test split, once, only if the §5 gate is met | ☐ **not delegated** |

Gate across the tree: `ruff` clean, `mypy --strict` clean (49 source files).
Gates on cycle-4 data: **20 passed, 4 skipped** (BAF still not vendored).

---

## Three findings. All committed; none depends on the rescore landing.

**1. Time-to-detection was never measurable** (`LIMITATIONS.md` §8.7a). `detection_rate_d7`
/ `d14` / `d30` read 0.000 for every rung *and every floor* in cycle 3 because drift onsets
were confined to days 30–240 while the validation window opens on day 240. Verified against
the committed ground truth: 294 fraud merchants, onset max 217. d7 needed onset ≥ 233 —
**0 of 294 qualified**. An oracle scores 0.000 too. §8.7 is preserved unedited with a
pointer; the numbers stand, the conclusion drawn from them does not.

**2. `volume_rank` wins on exposure, not on detection** (`LIMITATIONS.md` §8.3a). At the
same K, Rung 3 beats `volume_rank` on precision@K (0.869 vs 0.571) **and** recall@K (0.315
vs 0.195) **and** calibration (ECE 0.0077 vs 0.4866) and still loses 27% on savings. No
ranking-quality hypothesis produces that. `capacity.py` already ranks by expected rupees;
it was ranking on `p_declared_monthly_gmv` — the *declared* figure, corrupted by the
generator at σ = 0.55 — while `volume_rank` ranks on observed GMV and `true_loss` is
`loss_fraction × post-onset realised GMV`. Against realised loss: declared ρ = **0.533**,
observed ρ = **0.935**. At K = 15, exposure alone captures 20.5% of total loss when
declared, **42.7%** when observed, against an oracle's 46.2%. Reproduce with
`scripts/exposure_diagnostic.py`.

**Every savings verdict in §8.2–§8.5 was rendered through the weaker estimator**, including
the cut of Rung 4. Not overturned — provisional, in a way they were not previously reported
as being. The rescore is what settles them.

**3. §6's demo-premise falsification does not reproduce** (`LIMITATIONS.md` §6a). §6 records
the raw detector GREEN at +1.27pp and concludes the premise the system was built on does not
hold — a claim cited by the risk register below and by §8.2's K-1 reasoning. Measured on the
current gate: raw **+7.07pp RED**, residual **+2.70pp RED**, residual's advantage **+4.37pp**
rather than +0.55pp. So the raw detector *does* fail and the residual cuts the excess by 62%,
which is evidence **for** the premise, not against it.

The comparison is controlled: `scripts/g5_cycle_comparison.py` runs the gate's own functions
over both configs at the same seed and population, the two columns come out **bit-identical**,
and its cycle-4 column reproduces the live gate exactly. That also retires the
`docs/gates/GATES-CYCLE4.md` observation — there is no cycle-3 → cycle-4 effect in G5 at all.
**Not claimed: that §6 was wrong when written.** Its numbers carry no population size and
T-0101 moved the horizon 180 → 365 days underneath it. **Both detectors are still RED** and
§6a does not soften that.

---

## Next action

**Step 6 — rescore the full ladder on cycle-4 data, both exposure arms, five seeds.**

```
make features                      # ~70 min at --workers 10; the long pole
make gates                         # G1-G5 must be green before any model trains
uv run python -m rakshak.cli train --rung 2 --seed 42   # ... 43 44 45 46, then rung 3, 4
uv run python -m rakshak.cli eval  --rung R --seed S --exposure declared
uv run python -m rakshak.cli eval  --rung R --seed S --exposure realised
```

`--exposure declared` is the default and reproduces cycle 3's wiring exactly
(`test_the_wrapper_is_a_no_op_when_the_exposures_agree` asserts byte-identity). Arm B is
the pre-registered comparison. **Both arms, every rung, or the comparison is not
controlled.**

---

## The three things that must not slip

1. ~~EVAL-LOCK written before any model trains~~ — done, twice. `eval_module_sha256` is
   `c009e38d…` in **both** the cycle-3 and cycle-4 locks, byte-identical. **If that hash
   ever differs, the cycle's central claim is broken and the pre-registration is void.**
2. **The test split opens exactly once**, in step 8, and only if all four conditions in
   `PRE-REGISTRATION-CYCLE4` §5 hold. It has been opened **zero** times.
3. `cli.py` must call **both** `require_unlocked_or_refuse(split)` and `verify_lock()`
   before any scoring path. The primitive refuses on anything but the literal string `"1"`.

---

## Declared in advance, so it cannot be discovered conveniently later

- **The spec's test-fold yield estimate was wrong and the correction is pre-registered.**
  `12-spec-cycle4.md` predicted 13 in-window onsets in the test fold; measured 9.0, of which
  6.4 resolve. Confirmed analytically, not seed noise.
- **R2 and R9 — 25% of the fraud mix — have zero probability of onsetting in either
  evaluation window.** The affine rescale preserves relative position, so only R6 reaches
  day 364. R2 is the slow-ramp bust-out *v1 failed on*. Per-typology latency for R2 and R9
  is structurally uncomputable in cycle 4 and **must be reported as absent, not as zero.**
  Not patched around: widening those windows makes two typologies easier than their peers,
  which the spec rejected.
- **Floors are priced REVIEW-only (₹250/error); rungs are priced on their own actions,
  which may HOLD (₹8,250/error).** A 33× asymmetry. `savings_of_ranking`'s docstring claims
  the comparison differs only in the score vector — true floor-vs-floor, false
  floor-vs-rung. Fixing it means editing the locked eval package, so it is named and the
  hash is left alone.
- **The cycle-3 ladder was single-seed.** Every four-decimal number in it, including the
  0.6017 floor cycle 4 is measured against, is weaker than it looks. Cycle 4 scores all five.
- **INSEPARABLE is a pre-declared, acceptable outcome for Rung 9.** ~10 evaluable merchants,
  ±13 pp standard error. A rung that does not clear its gate is reported as not adopted and
  is **not tuned to rescue it.**

---

## Carried defects with no owner

**1. `tests/unit/test_cohort.py::test_what_the_cohort_residual_actually_does_under_p2` is
RED and was red before cycle 4 began.** Measured 0.3677 against a bar of `raw × 0.85` =
0.3647 — a 14.3% alert-rate reduction against a claimed >15%. Identical numbers before and
after the cycle-4 config change, verified by stashing. The threshold was set on a different
population and has drifted. **Not weakened to go green**; it needs either a re-derived
threshold or an acknowledgement that the claim is now 14%, and that is a decision, not a
patch.

**2. `configs/rung_roster.yaml` carries a `known_gap` and two `gap` fields saying
`ADR-V3-001` has no file.** It now does — `docs/adr/ADR-V3-001-no-autograd.md`. The roster
should cite that path when it is next regenerated.

**3. The stale deferral list.** `00-charter-v2.md` §8, `06-requirements-v2.md` §E and this
file's own predecessor all said Rungs 5–8 must not be started. GitHub #51 reversed that
explicitly and they were built. The three documents have not been updated to match.

---

## Session log pointer

`docs/logbook/T-*.md` — one file per ticket: built / surprised / broke. The surprise field
is the one that matters. The three most valuable, still:

- **T-111** — a flat `F_nb` over a *composed* intensity gave a realised Fano of 15.11,
  because the variance of the intensity adds to the count variance. Unit tests could not
  have caught it; they isolate the process at constant intensity. G1 did.
- **T-120** — parity stayed green while every baseline was empty. **Parity says two runners
  agree; it never says they agree about something meaningful.**
- **Cycle 4** — three literature surveys, run in parallel across quickest-change detection,
  delayed-feedback/PU learning and the cost/decision layer, and **none of them predicted the
  finding.** It came from reading `capacity.py` to answer a question one survey raised —
  does the decision layer rank by probability or by expected value — finding it already
  ranked by expected value, and realising that left exactly one place for the difference to
  live. The surveys were still worth their cost: 13c reached the same mechanism
  independently by tracing source, which is what turned one reading of the code into a
  finding two methods agree on.
