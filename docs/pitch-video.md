# Rakshak — pitch video: script, shot list, edit checklist

**Ticket:** T-0019. **Status:** script drafted before the freeze, per the ticket's own instruction
("the narrative does not depend on the verdict — only the numbers do"). **Numbers finalised** off
`README.md` as committed at `7fc2b77` (T-0013) plus `results/verdict.md`, `results/reasons.json`,
`results/calibration_gap.md`, `STATE.md`, `docs/adr/ADR-0003-no-reinforcement-learning.md`,
`docs/adr/ADR-0009-k1-label-informed-hmm.md`, and `docs/ARCHITECTURE.md`. **Every number below
carries the file it came from — see the provenance table at the end.** Do not swap a number in
this document without re-checking it against a committed file first.

**Not done by this ticket, deliberately:** the recording itself. That is a separate, human,
non-delegable task. This file is what makes recording possible.

---

## The script

Read-aloud target: **≤ 5:00.** Word count: **671 words.** At a narration pace of 140 words/minute
(the right assumption for text this numeric — digits read slower than prose) that is **4:48**; at a
brisk 150 wpm it is **4:28**; at a deliberately slow 130 wpm it is **5:10** — 10 seconds over, which
is exactly what the drop order below exists to absorb (see **Edit checklist → cut list**). The hero
claim (`00-charter.md` §2) lands at the **44-second mark** at 140 wpm, inside the ticket's 45-second
requirement.

Six beats, in the ticket's order, none reordered or dropped from the draft (beat 5 is marked as the
cut-first candidate for the *recording*, not removed from the *script*).

> Numbers in brackets are timestamps at the 140-wpm baseline, cumulative from 0:00.

### Beat 1 — The gap (0:00–0:44)

> Razorpay's Vulcan scores every transaction in milliseconds. Bumblebee reviews every merchant
> once, at onboarding. Nothing watches a merchant who already cleared drift from good to bad in
> the weeks after — bust-outs, laundering, refund collusion surface only when chargebacks land,
> forty-five to one-twenty days later. We pre-registered one falsifiable claim against that gap:
> Rakshak beats a static rule engine by twenty percent or more, relative, on the Bahnsen savings
> score, at equal analyst-hour budget, on merchants the model never saw — swept across the full
> cost-asymmetry range, with the point it fails stated out loud. That's the whole video: did it
> hold.

### Beat 2 — The mechanic (0:44–1:17)

> Here's how. For every merchant, Rakshak runs a hand-written Hidden Markov Model over their
> transaction stream — forward-backward, Viterbi, Baum-Welch, written in numpy, not a library.
> Each window updates a belief: a distribution over four latent states — healthy, ramp, fraud,
> dormant — conditioned only on that merchant's own past, never the future. That belief crosses one
> wire into the decision layer, which takes the cost-minimizing action — pass, review, or hold —
> under a fixed analyst-hour budget.

### Beat 3 — The explanation, centerpiece (1:17–2:27)

> Merchant M-zero-zero-one-three-one. This is the Viterbi decode — the same inference, not a
> report written afterward — reading straight from `results/reasons.json`, word for word:
>
> "On 26 August 2026 this account moved into a pattern we flag as sustained-abnormal-activity. The
> measurements that moved it there, each compared against this account's own trading history
> rather than against other merchants: share of repeat payers ran far above this account's own
> baseline, plus eight point one eight standard deviations; overlap between this week's payers and
> last week's ran far above baseline, plus five point four one; share of first-time payers ran far
> below baseline, minus six point eight four. What would resolve this: settlement-level invoices
> for the flagged period, and contact details for the payers behind the largest transactions."
>
> That's question three answered: when this merchant calls and shouts, this is what you read them
> — not a SHAP plot bolted on after the fact, but the state the model already believed, decoded
> jointly with the date it changed.

### Beat 4 — Honest measurement, the section that wins Track 02 (2:27–3:49)

> Now the section that wins this track — how we measured, not what we built. Kill criterion K1:
> does the HMM recover the four latent states? Unsupervised, it scored zero point zero nine one,
> against an oracle ceiling of zero point four zero four — parameters read straight off ground
> truth. Unreachable, because ramp, our early-warning state, sits one point one nine standard
> deviations from healthy, ninety percent of every window. The state the product exists to catch
> overlaps the most.
>
> We added label supervision. Every headline metric roughly doubled — and recall on ramp got
> worse: zero point three two eight down to zero point two three four. Supervision helps rare
> states that are separable, not rare states that overlap. We pre-registered a ramp-recall bar of
> zero point three five before measuring. It failed, and ships today as a strict, permanent xfail.
>
> And the headline claim: K2 FAILED. At the cited central asymmetry, Rakshak beats the rule engine
> by five point nine percent — against the twenty percent bar pre-registered before the sweep ran.
> Across the full range, it never crosses twenty. We report that, not around it.

### Beat 5 — What was rejected, and why (3:49–4:18) — **first cut on overrun**

> Why not reinforcement learning? This is genuinely a POMDP — hidden state, belief as sufficient
> statistic, three actions, a capacity budget. Rejected for two reasons: no reward signal inside a
> four-day build, chargebacks land forty-five to one-twenty days late; and training inside our own
> simulator would only teach the model our own assumptions back. We use Bayes Minimum Risk instead
> — closed form, one-step optimal, no training.

### Beat 6 — The limitation (4:18–4:48)

> One limitation, in full: sequence-layer metrics are measured on synthetic merchant streams with
> injected typologies; the generator is in this repo. The decision layer is additionally validated
> on BAF, a public benchmark derived from real bank data. Five of eight marginals diverge from a
> real transaction stream by close to two times or more, four by five times or more — one gap is
> structural, closed by no parameter choice.

---

## Shot list

One row per beat. **Screen sources are committed files only — nothing on screen is retyped into a
slide.** Where a number's canonical home is a regression-locked test rather than a `results/`
table (K1, beat 4's first paragraph — see the provenance table's note), the shot list says so
explicitly rather than implying it came from `make eval`.

| Beat | Duration | On screen | What's said | Artifact the on-screen number comes from |
|---|---|---|---|---|
| 1. The gap | 0:44 | `00-charter.md` §1–§2 open in an editor, scrolled to the pre-registration amendment (dated 2026-08-28, before T-0007b) | The gap, then the pre-registered claim, verbatim in substance | `00-charter.md` (committed, §1 and §2) |
| 2. The mechanic | 0:33 | `docs/ARCHITECTURE.md` §2's mermaid diagram (the four-layer flowchart), panning from generator → feature layer → HMM → decision layer as each is named | Belief over four latent states, forward-only filtered posterior, one wire into the decision layer | `docs/ARCHITECTURE.md` §2 (committed diagram; no fabricated animation — the diagram *is* the shot) |
| 3. The explanation | 1:10 | `results/reasons.json`, scrolled/grepped to the `M00131` entry, on screen long enough to read while narrated | The Viterbi-derived reason string, read word for word | `results/reasons.json` (golden-file tested, `tests/test_reasons.py`) |
| 4. Honest measurement | 1:22 | Three sequential shots: (a) `docs/adr/ADR-0009-k1-label-informed-hmm.md` for the K1 ARI/ceiling numbers and the RAMP-recall regression; (b) the `xfail(strict=True)` assertion in the test file naming the 0.35 bar; (c) `results/verdict.md`'s K2 table and verdict line | K1 gate, oracle ceiling, RAMP regression, the pre-registered xfail, K2 FAIL at +5.9% vs ≥20% | (a)+(b) `docs/adr/ADR-0009-k1-label-informed-hmm.md` and `tests/test_hmm_recovery_fullscale.py` — **regression-locked, re-verified by `make test`/`pytest`, not a `results/` table `make eval` writes; the K1 numbers are not regenerated by `make eval`, and the on-screen caption must say so.** (c) `results/verdict.md` |
| 5. What was rejected | 0:29 | `docs/adr/ADR-0003-no-reinforcement-learning.md`, scrolled to "Two facts kill it as build work" and the Decision section | The POMDP framing, the two disqualifying facts, BMR chosen instead | `docs/adr/ADR-0003-no-reinforcement-learning.md` (committed ADR) |
| 6. The limitation | 0:30 | `CLAUDE.md` non-negotiable #3 (the verbatim sentence, highlighted), cut to `results/calibration_gap.md`'s per-marginal diff table | The verbatim synthetic-data disclosure, then the 5-of-8 / 4-of-8 / structural-gap figures | `CLAUDE.md` (non-negotiable #3) and `results/calibration_gap.md` |

**T-0014 note.** The ticket allows `results/` artifacts *or* T-0014's viewer as screen sources. As
of this draft, **T-0014 (the results dashboard) does not exist in this repo** — `dashboard/` is
absent from the working tree and from `git ls-tree HEAD`. The shot list above is built entirely on
raw committed files for that reason, and does not assume the dashboard will land before recording.
If T-0014 ships before Wed 2 Sep, the shot list may substitute its rendered views for the equivalent
raw-file shots above **without changing any number or beat** — update this table's "On screen"
column only, not the script.

---

## Edit checklist

### Recording setup

- **Resolution:** 1920×1080, 30fps minimum. Record at the target resolution, not down-scaled later —
  terminal text legibility is the constraint, not final file size.
- **Audio:** separate mic track from screen capture; no laptop-mic room echo. Normalize to -16 LUFS
  for consistent playback loudness against other submissions in the review queue.
- **Terminal font size:** minimum 18pt at 1080p for any terminal/editor pane that stays on screen
  longer than 3 seconds (beats 3, 4, 5, 6 all hold a file on screen — this is most of the video).
  Verify legibility by watching a played-back capture at actual size before the final take, not by
  eyeballing the live recording window.
- **`make eval` on camera — do NOT claim it runs green.** T-0021 (verifying a clean checkout) has
  not run as of this draft. If `make eval` appears on screen at all (optional — it is not in the
  shot list above), the on-screen/spoken framing must be limited to "this generates every number in
  this video" and must **not** assert or imply a clean, currently-verified pass. Add the claim only
  after T-0021 closes; until then, prefer the static file shots in the table above, which carry no
  such dependency.

### Cut list, target timings, and drop order

| Beat | Target | Cumulative | Cuttable? |
|---|---|---|---|
| 1. The gap | 0:44 | 0:44 | No — hero claim, hard 45s requirement |
| 2. The mechanic | 0:33 | 1:17 | No |
| 3. The explanation | 1:10 | 2:27 | No — the centerpiece, Question 3 |
| 4. Honest measurement | 1:22 | 3:49 | No — "the credibility, not the caveat" |
| 5. What was rejected | 0:29 | 4:18 | **Yes — first and only cut on overrun** |
| 6. The limitation | 0:30 | 4:48 | No — required disclosure |

**Hard ceiling: 5:00.** At the 140-wpm baseline the full script lands at 4:48, 12 seconds of
margin. If the actual take runs slower (measured live at read-through, or if a slower ~130 wpm
delivery is used for clarity on the numeric beats), **drop beat 5 in full** before touching beat 4
— beat 5 is 29 seconds, more than enough margin even at a 130 wpm read (which would otherwise land
at 5:10). **Never shorten beat 4 to make room** — the K1 story and the RAMP regression are the
ticket's explicit "must NOT soften" items, and beat 6's verbatim disclosure sentence may not be
paraphrased or shortened either. If cutting beat 5 alone is insufficient, the next reduction is
tightening beat 1's non-hero-claim sentence (the "who hurts" framing before the pre-registered
claim), never the claim sentence itself.

### Export settings and final check

- Export H.264, target bitrate that keeps the terminal text artifact-free on the file-viewer shots
  (beats 3–6 are mostly static text — a lower bitrate than a typical talking-head video is fine here,
  the risk is over-compression turning small monospace text illegible, not motion blur).
- **Final check before submission — mandatory, not optional:** play back the finished cut once with
  the script open beside it, and for every number spoken or shown on screen, confirm it against the
  provenance table below. A number that appears in the final cut and is not in this table is a
  defect per the ticket's own "must NOT do" list and blocks submission until traced or removed.

---

## Provenance — every number in the script, traced

| Number spoken | Source file | Confirmed by |
|---|---|---|
| Vulcan scores every transaction; Bumblebee reviews once; 45–120 day chargeback lag | `00-charter.md` §1, `CLAUDE.md` (What this is) | direct read |
| The pre-registered claim: ≥20% relative, Bahnsen savings, equal analyst-hour budget, temporally-and-group-split held-out set, full asymmetry range swept | `00-charter.md` §2 (as amended 2026-08-28, T-0017) | direct read, quoted in `results/verdict.md` line 62 |
| Four latent states: healthy, ramp, fraud, dormant | `docs/ARCHITECTURE.md` §1, §3 (Layer 3) | direct read |
| Forward-only filtered posterior, never smoothed; hand-written HMM (forward-backward, Viterbi, Baum-Welch), numpy, log space | `docs/ARCHITECTURE.md` §3 (Layer 3); `CLAUDE.md` stack table | direct read |
| Pass/review/hold under fixed analyst-hour budget | `docs/ARCHITECTURE.md` §3 (Layer 4) | direct read |
| M00131 reason string, verbatim, incl. +8.18 / +5.41 / −6.84 SD | `results/reasons.json` → `reasons[]` entry `merchant_id: "M00131"` | `python -c` read of the file, byte-for-byte match confirmed |
| 56 flagged / 15 truly bad / 41 healthy / 3 Viterbi-disagreed (referenced in shot-list framing, not spoken) | `results/reasons.json` → `counts` | direct read: `{"merchants_in_split": 100, "merchants_truly_bad": 20, "merchants_flagged": 56, "flagged_and_truly_bad": 15, "flagged_and_healthy": 41, "viterbi_disagreed_with_flag": 3}` |
| K1: ARI 0.091 vs oracle ceiling 0.404 | `README.md` "The K1 story" section; `docs/adr/ADR-0009-k1-label-informed-hmm.md`; `STATE.md` "The K1 story" | direct read, consistent across all three; regression-locked by `tests/test_hmm_recovery_fullscale.py`, **not** a `results/` table `make eval` writes |
| RAMP sits 1.19σ from HEALTHY, ~90% of windows | `README.md` "The K1 story"; `STATE.md` line 338 | direct read |
| Supervision: RAMP recall 0.328 → 0.234 while doubling headline metrics | `README.md` "The K1 story"; `STATE.md` lines 354–360 | direct read |
| Pre-registered RAMP-recall ≥0.35 bar, failed at 0.234, strict `xfail` | `STATE.md` lines 362–363 | direct read |
| K2 FAIL: +5.9% relative at central asymmetry 13.1, vs ≥20% bar; never crosses 20% across 0.7–146.9 | `results/verdict.md` line 73 ("K2 VERDICT: FAIL... 5.9% relative"), line 134 (">=20% claim holds at no swept asymmetry between 0.7 and 146.9") | direct read of `results/verdict.md` |
| POMDP framing: hidden state, belief as sufficient statistic, 3 actions, capacity budget; no reward signal (45–120 day lag); training inside own simulator learns own assumptions; BMR chosen, closed-form, one-step-optimal | `docs/adr/ADR-0003-no-reinforcement-learning.md` (Context, Options, Decision) | direct read |
| Synthetic-data disclosure sentence, verbatim | `CLAUDE.md`, non-negotiable #3 | direct read, reproduced word-for-word in the script |
| 5 of 8 ratio-scale marginals diverge ≥1.9x, 4 of 8 diverge ≥5x; structural Fano-factor gap | `results/calibration_gap.md` ("Measured: of the 8 ratio-scale marginals, 5 diverge by 1.9x or more and 4 by 5x or more... One divergence is structural, not parametric") | direct read |

**Every number traces to a committed file.** The one deliberate exception to the "regenerable by
`make eval`" phrasing is the K1 pair (0.091 / 0.404) and the RAMP-recall pair (0.328 / 0.234): these
are regression-locked assertions re-verified by the test suite (`make test` / `pytest`), not rows a
`make eval` run writes to `results/`. The shot list and the "final check" step above both flag this
explicitly rather than letting it pass as an ordinary `results/` number — the ticket's own words
apply to the spirit of the constraint ("regenerable by `make eval`" is a proxy for "regenerable and
committed," and this pair satisfies the underlying requirement while missing the letter of "in
`results/`").

---

## Done-when self-check

- [x] Script exists, timed to ≤5:00 read aloud (671 words, 4:48 at 140 wpm, 4:28–5:10 across a
  130–150 wpm range, with an explicit drop-order absorbing the slow-end overrun)
- [x] Hero claim (`00-charter.md` §2) lands within the first 45 seconds (44s at baseline pace)
- [x] Every number in the script carries the artifact it came from (provenance table above)
- [x] RAMP regression (0.328 → 0.234) and K2's verdict (FAIL, +5.9% vs ≥20%) both appear, stated
  plainly, not softened, not edited around
- [x] Shot list maps every beat to a screen source in `results/`, `docs/`, or `CLAUDE.md` — noted
  explicitly where T-0014's viewer does not yet exist to serve as an alternative
- [x] Edit checklist names the drop order for overrun (beat 5 first, decided here rather than in
  the edit bay)
- [x] Synthetic-data framing spoken in `CLAUDE.md`'s verbatim wording (beat 6)
- [x] No claim that `make eval` runs green on camera (edit checklist, Recording setup)
