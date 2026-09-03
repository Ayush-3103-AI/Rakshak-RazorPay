# Rakshak G3 — pitch video: script, shot list, provenance, edit checklist

**Status:** script written against the **cycle-4 committed artifacts**, after the ladder was
scored and the cost sweep was run. This supersedes [`../../docs/pitch-video.md`](../../docs/pitch-video.md),
which is the **G2** script: its numbers are G2's, they are immutable under Prime Directive 2, and
**they must not be spoken over G3 footage.** If you have already recorded from that file, re-record
beats 3–5.

**Not done by this file, deliberately:** the recording. That is a human, non-delegable task. This
file is what makes recording possible in one sitting.

**Every number below carries the committed file it came from — see the provenance table in §4.**
Do not swap a number in this script without re-checking it against that file first. This is the
same rule the repo applies to itself, and the video is the one artifact where a stale number
cannot be caught by a test.

---

## 1. The script

Read-aloud target: **≤ 5:00.** Word count: **684 words.** At 140 words/minute — the right pace for
text this numeric, since digits read slower than prose — that is **4:53**; at a brisk 150 wpm,
**4:34**; at a deliberate 130 wpm, **5:16**, which is what §5's cut list exists to absorb.

Six beats. Timestamps are cumulative at the 140-wpm baseline.

### Beat 1 — The gap (0:00–0:35)

> Razorpay's Vulcan scores every transaction in milliseconds. Bumblebee reviews every merchant
> once, at onboarding. Nothing watches a merchant who already cleared drift from good to bad in
> the weeks after — bust-outs, laundering endpoints, refund collusion surface only when the
> chargebacks land, forty-five to a hundred and twenty days later. Rakshak is the sentinel for
> that gap: every day, for every cleared merchant, one of pass, review or hold, under a hard
> analyst-capacity budget, with a merchant-readable reason on every non-pass.

### Beat 2 — What we actually built, and in what order (0:35–1:24)

> Here is the part that matters. We did not build a model and then measure it. We built the
> generator, the split engine, the metric suite, the four floors and the cost model **first** —
> then hashed all of it into a lock file, before the first model existed. Four locks, superseding
> forward. The test split carries an open counter, and `make eval` refuses to touch it unless an
> environment variable is set that is not set anywhere in this repository. Only then did we race a
> ladder of sixteen policies against it, at five seeds each, on forty thousand merchants over
> three hundred and sixty-five days — a hundred and twenty-two million transactions.

### Beat 3 — The result, and the floor that nearly beat us (1:24–2:08)

> One row survives. Rung four, priced on realised exposure, beats every floor on every seed:
> point five nine eight one against the hardest floor's point five two four zero. That floor is
> `volume_rank` — rank merchants by size, no learning at all — and it is the hardest thing on the
> ladder, because on rupees an exposure estimator beats a fraud-probability ranker that is never
> told what is at stake. Three other policies flip with the seed: they beat the floors on exactly
> one seed of five. A single-seed ladder would have reported any of them as a win, to four
> decimal places.

### Beat 4 — Where the margin actually comes from (2:08–3:03)

> Then we swept it. Every savings number we published before last week was a point estimate at one
> guessed cost matrix. Across four orders of magnitude of false-hold-to-fraud-loss asymmetry, with
> nothing refitted, rung four holds between point five eight five three and point six zero zero
> one, and beats the floor at five of five ratios. Then we decomposed that margin — the part I
> would rather not tell you. Make `HOLD` unreachable and change nothing else: the margin over the
> floor falls from plus zero seven four zero to plus zero four zero three. Price it as a raw
> ranking, and **every rung loses to `volume_rank`** — the best pure rupee-ranker among them is
> rung one, the rule engine. The advantage is a decision-layer result, not a modelling one.

### Beat 5 — What we killed (3:03–3:51)

> Prime directive six: a rung that loses is a finding, not an embarrassment. Learned attention
> lost to the fixed pooling it was meant to replace. Segmentation-based onset localisation lost
> to the trivial rule "onset equals alert day", at both seeds, on every statistic. A neural
> intensity model made a circularity objection worse, exactly as predicted before it was built.
> Page-CUSUM's primary gate turned out not to be computable from the harness's output — and that
> decides it. All five are in the tree, in the roster, and in `LIMITATIONS.md` with numbers.
> Rung five has the best PR-AUC on the whole ladder and near-worst savings — a calibration
> problem, not a ranking one.

### Beat 6 — The door we did not open (3:51–4:53)

> And the pre-registered gate failed: zero of five seeds and zero of five sweep ratios cleared the
> bar. Part of why is that the bar was anchored to a cycle-three threshold this cycle's own
> regeneration invalidated. That is a pre-registration error, recorded as one rather than quietly
> re-anchored. So the test split stayed shut. `open_count` is zero.
> There is no held-out number in this project, and I am not going to tell you the result would
> probably have held. Every figure I have quoted is on validation, on synthetic merchant streams
> from the generator in this repo. The best row on my best table has a median time-to-detection of
> infinity. The dashboard reads every number off the committed artifacts at load time, so it
> cannot drift from what was measured — and `LIMITATIONS.md` is the longest document in the repo
> on purpose. Start there.

---

## 2. What the screen shows, beat by beat

| Beat | On screen | Source |
|---|---|---|
| 1 | The live panel's opening screen — the gap between Vulcan and Bumblebee | https://ayush-3103-ai.github.io/Rakshak-RazorPay/ |
| 2 | `EVAL-LOCK-CYCLE4.json` scrolled slowly: `eval_module_sha256`, `open_count: 0`. Then the terminal: `make eval` refusing the test split | [`../EVAL-LOCK-CYCLE4.json`](../EVAL-LOCK-CYCLE4.json) · `make eval` |
| 3 | The ladder table, arm B, with `volume_rank` highlighted as the row to beat | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) · panel |
| 4 | The sweep table — five ratios across the top, the floor as a flat line under it. Then Table A / Table C / Table B side by side | [`results/cost_sweep.md`](results/cost_sweep.md) §2–§5 |
| 5 | `configs/rung_roster.yaml` scrolled, then `LIMITATIONS.md` §§12–15 | [`../configs/rung_roster.yaml`](../configs/rung_roster.yaml) · [`../LIMITATIONS.md`](../LIMITATIONS.md) |
| 6 | The verdict block: `0/5 seeds`, `0/5 ratios`, `-> FAIL`. Hold on `open_count: 0`. End on the panel URL | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) §3, §3a, §6 |

Record the panel at 1920×1080, browser chrome hidden. The panel is a free-scrolling read across
eight screens; do not scrub it faster than it can be read, and do not zoom past a number you are
not saying out loud.

---

## 3. The three things the video must say out loud

These are not stylistic. They are the track's own bar and this repo's non-negotiables.

1. **Synthetic.** Say the word. The sequence-layer metrics are measured on synthetic merchant
   streams with injected typologies, and the generator is in this repo. BAF — the one external
   anchor — was validated in G2 and is **not vendored in the G3 tree**, so four of the twenty-four
   cycle-4 gates skip for that reason and every G3 number is synthetic-only.
2. **Validation, never test.** Every number spoken is on the validation split. `open_count` is 0.
3. **Defense-only.** The generator emits risk typologies to *test detection*. It is an evaluation
   artifact, not a fraud toolkit, and no real Razorpay data, API or internal system is touched.

---

## 4. Provenance — every number in §1

| Beat | Number | Committed source |
|---|---|---|
| 1 | 45–120 day chargeback delay | `configs/scenario_v2.yaml`; [`../../README.md`](../../README.md) |
| 2 | four locks, superseding forward | `EVAL-LOCK.json`, `-CYCLE2`, `-CYCLE3`, `-CYCLE4` |
| 2 | `open_count: 0` | [`../EVAL-LOCK-CYCLE4.json`](../EVAL-LOCK-CYCLE4.json); `artifacts/lock_state.json` |
| 2 | sixteen policies, 5 seeds | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) — 16 policy rows, `seeds [42..46]` |
| 2 | 40,000 merchants × 365 days, 121.9M transactions | `configs/scenario_v2.yaml`; [`../../README.md`](../../README.md) |
| 3 | rung 4 realised = **+0.5981**; per-seed `[0.5862, 0.5927, 0.5927, 0.6211, 0.5976]` | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) §3 |
| 3 | `volume_rank` floor = **+0.5240** | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) §3b |
| 3 | three policies flip with the seed | [`../../README.md`](../../README.md) §Results |
| 4 | sweep range **+0.5853 to +0.6001**, 5 of 5 ratios | [`results/cost_sweep.md`](results/cost_sweep.md) §5.1 |
| 4 | margin **+0.0740** as scored → **+0.0403** with HOLD unreachable | [`results/cost_sweep.md`](results/cost_sweep.md) §5.3 (ratio 0.01) |
| 4 | raw ranking margin **−0.2892**; best pure rupee-ranker is rung 1 | [`results/cost_sweep.md`](results/cost_sweep.md) §5.2, §5.3, Table B |
| 5 | five rungs not adopted — 8 (§12), 7b (§13), 5b (§14), 8b (§15), 9 (§9.6) | [`../LIMITATIONS.md`](../LIMITATIONS.md); [`../configs/rung_roster.yaml`](../configs/rung_roster.yaml) |
| 5 | rung 5: best PR-AUC, near-worst savings | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) ladder table |
| 6 | gate **0/5 seeds**, **0/5 ratios**, FAIL | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) §3, §3a |
| 6 | the gate was anchored to a cycle-3 threshold cycle 4 invalidated | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) §3b; [`PRE-REGISTRATION-CYCLE4-2026-09-01.md`](PRE-REGISTRATION-CYCLE4-2026-09-01.md) §5 |
| 6 | rung 4 `ttd_median_days` = `inf` | [`results/CYCLE4-VERDICT.txt`](results/CYCLE4-VERDICT.txt) ladder table |
| 6 | 4 of 24 gates skip, BAF not vendored | [`../project-context/STATE.md`](../project-context/STATE.md); [`../../README.md`](../../README.md) |

**One number deliberately not in the script.** `detection_rate_d30` became measurable this cycle —
the machine verdict says **13 of 16** policies score non-zero against 0 of 7 in cycle 3. It is a
real finding and it is on the panel, but the latency half of the result is powered by roughly
seven evaluable merchants, and a spoken number with a ±19 pp standard error behind it is a number
the video cannot caveat fast enough. It is left to `LIMITATIONS.md` §9.8, which can.

---

## 5. Edit checklist

Before recording:

- [ ] Re-read §4 against the committed files. Any mismatch is a stop, not a rounding.
- [ ] `git log -1` — record the commit the footage is taken at, and say it in the description.
- [ ] Open the live panel and confirm it renders; the numbers on it must match §4.

Cut list, in the order to drop if the read runs long — the first three lose no claim:

1. Beat 2's dataset dimensions (the panel shows them).
2. Beat 3's "to four decimal places".
3. Beat 5's rung-five PR-AUC/savings aside.
4. Beat 5 as a whole — **last resort.** Dropping it removes the negative results, which is the
   half of this submission the track's bar is actually about. Cut anything else first.

**Never cut:** the word *synthetic* (§3.1), `open_count: 0` (§3.2), or beat 4's decomposition.
Quoting +0.0740 without +0.0403 beside it is quoting the flattering number, and the repo says so
in writing before anyone asks.
