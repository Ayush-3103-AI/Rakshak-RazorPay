<h1 align="center">Rakshak</h1>
<p align="center"><b>Post-onboarding merchant risk sentinel</b><br/>
Razorpay AI Buildathon 2026 · Track 02 (AI Risk Manager) · solo build</p>

<h2 align="center">▶ START HERE — open the dashboard first</h2>

<p align="center">
  <a href="https://ayush-3103-ai.github.io/Rakshak-RazorPay/"><b>https://ayush-3103-ai.github.io/Rakshak-RazorPay/</b></a>
</p>

<p align="center">
  <b>A two-minute read, eight pages, one scroll each.</b><br/>
  It is the intended entry point for this submission and it answers, in order:<br/>
  what the gap is · where it sits in Razorpay's stack · what Rakshak decides ·<br/>
  what the result was · whether it survives stress · what we killed · how to check it.
</p>

<p align="center">
  Every figure on it is read at load time from the committed artifacts in this repo.<br/>
  Nothing on it is typed in by hand, so it cannot drift from what was measured.
</p>

---

> **Reviewers: please open the dashboard before reading further.** This README is the
> written version of the same argument, and it is long on purpose. The dashboard is the
> two-minute version, and it is the one built to be read first. Deeper technical evidence
> sits behind the **Full evidence** link in its top-right corner.
>
> **If the hosted link is unavailable**, run it locally — it is a static build over the
> committed artifacts and needs no backend, no keys and no network:
>
> ```bash
> cd v3/dashboard
> npm ci
> npm run dev        # then open the URL it prints
> ```

**Current tree:** [`v3/`](v3/) · **Honest failures, with numbers:** [`v3/LIMITATIONS.md`](v3/LIMITATIONS.md)

> **16 policies. 5 seeds. 4 sealed locks. One row survived.**
>
> `rung4` under realised exposure is the only policy on the ladder that beats **every** floor
> on **every** seed — and its margin holds from **+0.5853 to +0.6001** across four orders of
> magnitude of cost asymmetry. Roughly **45% of that margin is the decision layer, not the
> ranker**, and this document says so before you ask. Everything else we built, we killed on
> the record.

*Sequence-layer metrics are measured on synthetic merchant streams with injected typologies;
the generator is in this repo. The decision layer is additionally validated on BAF (Feedzai,
NeurIPS 2022), a public benchmark derived from real bank data.*

**Scope of that sentence.** BAF was vendored and validated in **G2**. It is **not vendored in
the G3 tree** — 4 of the 24 cycle-4 gates skip for that reason — so every G3 number below is
**synthetic-only**, on the **validation** split. The test split has never been opened.

---

## The gap

Razorpay's **Vulcan** scores every *transaction* in milliseconds. Razorpay's **Bumblebee**
reviews every *merchant* once, at onboarding. Nothing watches a merchant that already cleared
onboarding drift over the following weeks — so bust-outs, laundering endpoints, category drift
and refund collusion surface only when chargebacks land 45–120 days later.

Rakshak is the sentinel for that gap: every day, for every cleared merchant, one of
`PASS` / `REVIEW` / `HOLD`, under a hard analyst-capacity budget, with a merchant-readable
reason attached to every non-`PASS`.

---

## Three generations

Each one starts from the previous one's falsification, not its success.

| | | Thesis | What happened |
|---|---|---|---|
| **G1** | `ver1/` · separate tree, **not published** | A per-merchant rupee risk budget, spent by a constrained RL policy under a hard safety filter | Superseded by a different question, not a losing number. ₹13.96 L net value, 95.6% fraud prevented, 0 hard-limit violations — **cited, not recomputed**, and not independently checkable: G1 has no public repository |
| **G2** | this repo, root | A per-merchant HMM over the transaction stream, four latent risk states, decisions priced by cost | **The pre-registered claim failed**, and a uniform random score beat every fitted model on savings |
| **G3** | [`v3/`](v3/) | Fix the generator and the harness *first*, seal them, then race a ladder of policies against explicit floors | **The test split stayed shut.** One row survives every floor on every seed; four new rungs were built and all four NOT ADOPTED |

> **Naming.** These public labels exist because the internal vocabulary collides:
> `v3/project-context/00-charter-v2.md` calls G2 **"v1"** and G3 **"v2"**, while the separate
> G1 tree calls itself **"v1"**. Nothing in any charter, lock, ADR or results file was
> edited to introduce G1/G2/G3 — the mapping lives here and on
> [`v3/configs/journey.yaml`](v3/configs/journey.yaml), and travels on `journey.json`.

### What died at each boundary

- **G1 → G2.** Not a losing metric — a different unit of analysis. G1 asks "how much risk can
  this merchant afford *on this transaction*"; the problem is only visible at the
  merchant-week level, so G2 started from a new thesis and a new harness rather than tuning G1.
- **G2 → G3.** G2's own measurements falsified G2. The HMM lost to plain LightGBM by 0.3176
  PR-AUC, the pipeline cleared the rule engine by 5.9% against a self-imposed 20% gate, and the
  evaluation ran at 20% prevalence instead of a realistic ~1.5%. Diagnosis: until the generator
  is right, every model comparison is measuring the generator. G3 fixes the generator and the
  harness before any model is written.

Full G2 verdict, sweep and ceilings: [`results/verdict.md`](results/verdict.md) ·
ablations: [`results/ablations.md`](results/ablations.md) ·
BAF: [`results/baf_validation.md`](results/baf_validation.md).

---

## How to believe any of this

The differentiator here is not the model. It is what was fixed *before* the chart existed.

**The harness is sealed before the models are written.** Each cycle hashes the eval module, the
generator module and the scenario config into a lock file, records the freezing commit, and
carries an **open counter** for the test split. Four locks, superseding forward, all readable in
[`v3/EVAL-LOCK-CYCLE4.json`](v3/EVAL-LOCK-CYCLE4.json) and rendered on the panel.

**Claims are pre-registered, then reported either way.** Cycles 3 and 4 name a pre-registration
document written *before* the run that tested them
([cycle 4](v3/docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md)). Cycle 4's gate failed on **both**
conjuncts — and it failed partly because it was anchored to a cycle-3 threshold (0.7017) that
cycle 4's own regeneration invalidated (the real floor is 0.5240). That is recorded as a
**pre-registration error**, not re-anchored after the fact.

**Every policy is scored against explicit floors.** A model that beats nothing is not a result.
`all_pass`, `all_hold`, `random_at_k` and `volume_rank` are priced on every row, and a row that
loses is marked as losing to *that named floor*. `volume_rank` — rank merchants by transaction
volume, no learning at all — is the hardest floor on the ladder.

**`make eval` refuses the test split** unless `RAKSHAK_UNLOCK=1`. It is not set anywhere in
this repo.

---

## Results

**Validation split · cycle 4 · 5 seeds · K = 30 · 6,025 merchants · 361,500 scored
merchant-days.** Dataset: 40,000 merchants × 365 days, **121.9M transactions**, scored in
**215.3 s — 6.78× inside the NFR-10 budget**. Gates: 20 passed, 4 skipped (BAF not vendored).

**One row beats every floor on every seed:** `rung4_realised_exposure`. Three other policies
(`rung2`, `rung3`, `rung9` under realised exposure) **flip with the seed** — beating or failing
the floors on exactly one seed of five. A single-seed ladder, which is what cycle 3 was, would
have reported any of them either way with four decimals of apparent precision.

**The margin is not a point estimate.** Every savings figure this project published before the
cost sweep ran was a single point estimate at one cost matrix. Swept across four orders of
magnitude of false-hold-to-fraud-loss asymmetry, with nothing refitted:

| ratio | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| `rung4` (realised) | +0.5980 | +0.5981 | +0.5986 | **+0.6001** | +0.5853 |
| `volume_rank` floor | +0.5240 | +0.5240 | +0.5240 | +0.5240 | +0.5240 |

The shipped cost matrix sits at ratio **0.15398 — inside** the declared grid, which was *not*
extended after seeing the curve. Full tables: [`v3/docs/results/cost_sweep.md`](v3/docs/results/cost_sweep.md).

**Where that margin comes from, decomposed.** Make `HOLD` unreachable and change nothing else:
`rung4`'s margin over the floor falls from **+0.0741 to +0.0405**. It still wins at 5 of 5
ratios — but the pricing asymmetry the pre-registration disclosed is worth about **45%** of it.
Priced as a raw REVIEW-only ranking, **every rung loses to `volume_rank`**, and the best pure
rupee-ranker among them is Rung 1, the rule engine. **The advantage is a decision-layer result,
not a modelling one.**

Everything above, live and reading from the committed artifacts:
**https://ayush-3103-ai.github.io/Rakshak-RazorPay/**

---

## What we killed

Prime Directive 6: *a rung that loses is a finding, not an embarrassment.* All of these were
built, scored on real cycle-4 data, and dropped — with numbers, in
[`v3/LIMITATIONS.md`](v3/LIMITATIONS.md).

| | What it would have claimed | Why it was dropped |
|---|---|---|
| Rung 5b — learned attention | A fitted pooling parameter over payer capsules | Loses to the pooling it replaces |
| Rung 7b — segmentation onset | Better drift-onset localisation | Loses to "onset = alert day", at both seeds, on every statistic |
| Rung 8 — calibrated null | A distribution-free null for the alert score | The one part that does not survive measurement |
| Rung 8b — neural intensity | A learned temporal point process | Made the circularity worse, as predicted |
| Rung 9 — Page/CUSUM | Changepoint detection on within-day rank | Primary gate not evaluable — and that decides it |

Two more that are findings rather than rungs:

- **Rung 5 has the best PR-AUC on the ladder and near-worst savings.** A calibration problem,
  not a ranking one.
- **Time-to-detection was never measurable** in cycle 3. `detection_rate_d7/d14/d30` read 0.000
  for every rung *and every floor* because drift onsets were confined to days 30–240 while the
  validation window opens on day 240. An oracle scores 0.000 too. The numbers stand; the
  conclusion drawn from them does not.

---

## Reproduce

```bash
cd v3
uv sync
make all      # lint → parity → gen → gates → perf → test; must pass from a clean clone
make report   # regenerate docs/results_v2.md from the frozen eval
```

The dashboard is a static build over the same committed artifacts — no backend:

```bash
cd v3/dashboard && npm ci && npm run dev
```

`make all` passing **from a clean clone** is a stop-work condition, not a nicety: G1's single
largest disqualification risk was `make eval` not reproducing on a fresh checkout, so CI clones
the repo into a scratch path and builds from nothing on every push.

---

## Read next

| | |
|---|---|
| [`v3/LIMITATIONS.md`](v3/LIMITATIONS.md) | Every failure, with the number. The longest document in the repo, and deliberately so. |
| [`v3/README.md`](v3/README.md) | The G3 tree: prime directives, make targets, the unlock rule. |
| [`v3/project-context/STATE.md`](v3/project-context/STATE.md) | The resume point — what is open, what is closed. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The four-layer shape and the ADRs (G2). |
| [`results/`](results/) | G2's verdict, ablations, BAF validation, sensitivity, black-swan. |

**Defense-only.** The generator produces fraud typologies to *test detection*; it is an
evaluation artifact, not a fraud toolkit. No real Razorpay data, APIs or internal systems are
used anywhere.

Licence: [MIT](LICENSE).
