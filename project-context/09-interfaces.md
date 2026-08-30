# 09 — Interfaces

**Written 2026-08-30 under T-0020, from the code as it stands.** This file is descriptive, not
aspirational: every signature, field and unit below was read out of `src/` on that date. Where the
implementation is weaker than the contract someone might want, the file says so rather than
describing the contract someone might want.

## Authority

Until this file existed, the `Scorer` contract lived in **`src/rakshak/eval/harness.py`'s module
docstring**, and that is what T-0006b built against. That docstring is still correct and still
worth reading — it carries the rationale. **This file is now the authoritative statement of the
module boundaries**; the docstring is the reference implementation of the part it covers. If the
two ever disagree, the code wins and this file is stale — fix it, do not fix the code to match it.

Scope: the four boundaries tickets actually cross — `Scorer`, `Split`, `CostParams`,
`PolicyResult` — plus the `results/` artifact schemas T-0014 consumes. Everything else in `src/`
is internal and may change without a note here.

---

## 1. `Scorer` — `eval/harness.py`

```python
Scorer = Callable[[Split, np.random.Generator], "pd.Series | pd.DataFrame"]
```

A scorer answers one question: *how suspicious is each merchant in this split?*

**Signature.** `(split: Split, rng: np.random.Generator) -> pd.Series | pd.DataFrame`, indexed by
`split.merchant_ids`.

**What it may return.**

| Return | Meaning |
|---|---|
| `pd.Series` | Suspicion score per merchant. `flag_day` is filled with `NaN`. |
| `pd.DataFrame` with a `score` column | Same, as a frame. |
| the same frame plus a `flag_day` column | First day the model raised a flag; `NaN` if never. |

`harness._normalise` coerces either form to a `["score", "flag_day"]` frame reindexed onto
`split.merchant_ids`, and **raises `ValueError` if a DataFrame has no `score` column**. A missing
`flag_day` is not an error; it makes median detection lag uncomputable for that model, so any
time-resolved model should return one. All three fitted scorers in the registry (`rules`, `gbdt`,
`hmm`) return the frame form; `random` returns a bare Series.

**Units and ranges.**

- `score` — dimensionless suspicion, **higher means more suspicious**. Ranking metrics (`pr_auc`,
  `precision_at_k`) use it raw. The decision layer clips it to `[0, 1]` and consumes it **as if it
  were a calibrated posterior** (`harness.evaluate_model`). It is not calibrated: T-0008
  (empirical-Bayes shrinkage) was cut, so no recalibration happens anywhere in this repo. Under a
  rank-only policy that would cost only the Brier column; under Bayes Minimum Risk it moves the
  argmin. Every `savings` number in `results/` inherits that limitation.
- `flag_day` — an integer **calendar day** since `config.GENERATOR_START_DATE`, not a window
  index. A window-based model must add `metrics.WINDOW_ATTRIBUTION_OFFSET_DAYS`
  (`WINDOW_DAYS - 1`) at source, so the day it reports is the earliest day the flag could have
  been raised. This was the T-0011 lag defect; the convention is now uniform across the registry.

**Registration.** One line per model:

```python
MODEL_REGISTRY["rules"] = score_rules
```

`EXPECTED_MODELS` lists the models the frozen eval requires a row for. A name in `EXPECTED_MODELS`
but not in `MODEL_REGISTRY` is printed as **ABSENT**, never silently dropped — that is how `bocpd`
(T-0010, cut) appears in `results/summary.md`.

**Determinism (NFR-003).** Each scorer gets its own generator, seeded from
`(seed, zlib.crc32(name))`, so adding a model never perturbs another model's numbers. A scorer must
draw randomness only from the `rng` it is handed.

**The seam the contract does not carry.** `Scorer` takes no dataset argument, and
`gbdt.fit` / `hmm_score.fit` call `load_split("train")` **themselves**. Alternate datasets are
therefore selected by `eval.splits.active_dataset()` — module-level state entered by
`harness.run()` — not by an argument. The consequences, and the one way to misuse it, are in the
2026-08-30 amendment to `11-tickets/T-0022b.md`. Never call a `fit()` directly with a hand-built
split; `tests/test_dataset_seam.py` exists to catch exactly that.

---

## 2. `Split` — `eval/splits.py`

A frozen dataclass. One evaluation split is **a merchant group crossed with a time window**, and
this module is the single place that decides which rows any model may see.

| Field | Type | Meaning / units |
|---|---|---|
| `name` | `str` | `"train"`, `"validate"` or `"test"` |
| `start_day` | `int` | Window start, days since `GENERATOR_START_DATE`, inclusive |
| `end_day` | `int` | Window end, exclusive |
| `merchant_ids` | `tuple[str, ...]` | Sorted. **Disjoint across splits.** |
| `transactions` | `pd.DataFrame` | This split's merchants, all rows with `day < end_day` — history included. Carries an integer `day` column. |
| `labels` | `pd.Series` | 1 if the merchant enters a state in `BAD_STATES` before `end_day`. Indexed by merchant_id. Dimensionless. |
| `transition_day` | `pd.Series` | Day of first bad state, `NaN` if never. Units: days. Ground truth for detection lag. |
| `transition_timestamp` | `pd.Series` | Same, as a timestamp. `NaT` if never. |
| `loss_inr` | `pd.Series` | `L_m` — **realised** loss, `r_cb * (1 + phi) * gross volume while bad`. Units: INR. 0 for healthy merchants. |
| `value_inr` | `pd.Series` | `V_m` — expected remaining **lifetime** gross margin, `g * v_m * l_m`. Units: INR. |

Properties: `window_transactions` (rows inside the window only), `n_merchants`, `prevalence`.

**Guarantees `load_split` enforces, in code, on every call.**

1. **Merchant-group disjointness (NFR-002).** `assert_no_leakage` runs on every load. Group
   assignment is deterministic and **does not depend on `--seed`** — the frozen eval must not move
   when someone changes a seed — and is stratified by typology, interleaving sorted merchant IDs
   3:1:1 so every split sees all five typologies (FR-018 needs per-typology reporting).
2. **Temporal disjointness.** `SPLIT_DAY_BOUNDS`: train `[0, 180)`, validate `[180, 210)`, test
   `[210, 270)`. `assert_window_is_frozen()` pins them.
3. **The test-window lock.** `load_split("test")` raises `PermissionError` unless `unlock_test=`
   is one of `{"T-0011", "T-0013"}` — the only tickets 06-requirements.md §3 authorises to open
   it. This is the structural version of "test set touched exactly once, at the end".
   **A dataset override is not a test-window unlock**; the lock applies whichever dataset is
   active.

**The one thing that looks like leakage and is not.** `Split.transactions` carries each merchant's
history from day 0 up to the window end, not just the rows inside the window: a per-merchant
sequence model has to see that merchant's own past to hold a belief about it. Those rows were never
in another split and are strictly earlier than the decision point. Use `window_transactions` when
you want the window rows alone.

---

## 3. `CostParams` — `decision/policy.py`

Per-merchant cost primitives for one scoring run. Defaults are `config.py`'s shipping central
values, so `CostParams(loss, value)` reproduces `eval.metrics.action_cost` exactly.

| Field | Symbol | Units | Default |
|---|---|---|---|
| `loss_inr` | `L_m` | INR, shape `(n,)` | required |
| `value_inr` | `V_m` | INR, shape `(n,)` | required |
| `cost_review_inr` | `c_rev` | INR per REVIEW | `COST_REVIEW_INR` |
| `p_analyst_miss` | `p_miss` | probability a review clears a truly-bad merchant | `P_ANALYST_MISS` |
| `residual_leakage_rho` | `rho` | share of loss still leaking after a HOLD | `RESIDUAL_LEAKAGE_RHO` |
| `p_churn_given_hold` | — | P(churn given a wrongful HOLD) | `P_CHURN_GIVEN_HOLD` |
| `cost_support_inr` | `c_support` | INR per HOLD escalation | `COST_SUPPORT_INR` |
| `tau_review_hours` | `tau` | **hours** per review | `TAU_REVIEW_HOURS` |
| `fp_cost_scale` | — | dimensionless multiplier on the whole FP branch | `1.0` |

Derived property: `fp_cost_inr` = `fp_cost_scale * (p_churn_given_hold * V_m + c_support)`,
units INR.

**`fp_cost_scale` is the only field the FR-020 sweep moves.** That is deliberate: it varies the
*asymmetry* without varying the absolute size of fraud loss along with it. Every other field is
parameterised so a caller *can* vary it, not because anything in this repo does.

**`L_m` and `V_m` are the definitions T-0007a corrected.** `V_m` is the platform's own ~10 bps
lifetime margin, not the merchant-facing 2% MDR — a price is not a margin. `L_m` is realised
chargeback loss, not gross turnover while bad. The previous pair was wrong by roughly 15x on `L_m`
and 3x net on `V_m`, in opposite directions, which is why no sanity check on their ratio alone
could ever have found it.

---

## 4. `PolicyResult` — `decision/policy.py`

Returned by `bmr_policy(p_bad, params, capacity_hours) -> PolicyResult`: unconstrained Bayes
Minimum Risk over `{PASS = 0, REVIEW = 1, HOLD = 2}`, then the capacity constraint.

| Field | Type | Meaning / units |
|---|---|---|
| `actions` | `np.ndarray`, shape `(n,)` | Action per merchant, ints; names in `ACTION_NAMES` |
| `expected_costs` | `np.ndarray`, shape `(n, 3)` | Column order **PASS, REVIEW, HOLD**. Units: INR. |
| `n_reviewed` | `int` | REVIEWs **after** the constraint |
| `n_held` | `int` | HOLDs. Consumes no analyst hours. |
| `hours_used` | `float` | `tau * n_reviewed`. Units: hours. **Never exceeds `capacity_hours`.** |
| `capacity_hours` | `float` | `B`, the analyst-hour budget for the period. Units: hours. |
| `review_slots` | `int` | `floor(B / tau)` — the budget expressed in merchants |
| `binding_constraint` | `str` | `"capacity"` or `"none"` |
| `n_downgraded` | `int` | REVIEWs the constraint forced to their next-best action |
| `unconstrained_n_reviewed` | `int` | REVIEWs BMR would have chosen with no budget |

**`binding_constraint` is a reported field, not an inferred one (FR-017).** `"capacity"` means the
budget forced at least one downgrade; `"none"` means unconstrained BMR already fitted inside it.
A reader must never have to derive this by comparing `hours_used` against `capacity_hours` — a
policy can consume its budget almost exactly and still not have been constrained by it. The gap
between `unconstrained_n_reviewed` and `n_reviewed` is the capacity story this project exists to
tell, so both are carried, not one.

**Which REVIEWs survive the constraint.** The ones with the largest *regret* — the best non-REVIEW
alternative minus REVIEW. Downgrading the smallest regrets first is exactly optimal here because
every review costs the same `tau`. A downgraded merchant may still be HELD, because HOLD consumes
no analyst hours; that is the honest reading of the constraint and it is why `n_held` sits beside
`n_reviewed` rather than being folded into it.

`eval/oracle.py`'s `OracleResult` is the parallel shape for hindsight ceilings — `name`, `actions`,
`n_reviewed`, `n_held`, `hours_used`, `loss_averted_inr` (INR), `savings` (dimensionless),
`capacity_binding` (bool). It is a separate type on purpose: an oracle is not a policy.

---

## 5. `results/` — the artifacts T-0014 consumes

Everything here is produced by `make eval` at a fixed seed and is git-tracked. **A dashboard reads
these files and nothing else** — T-0014's L3 gate forbids mock data and forbids a silent fallback
to a hardcoded number when an artifact is missing.

| Artifact | Producer | Format |
|---|---|---|
| `summary.md` | `rakshak.eval.harness` | Markdown, `validate` window |
| `sensitivity.md` | `rakshak.eval.harness` | Markdown, FR-020 |
| `sensitivity.csv` | `rakshak.eval.harness` | **CSV — the machine-readable sweep** |
| `figures/sensitivity.png` | `rakshak.eval.harness` | PNG |
| `verdict.md` | `rakshak.eval.verdict` | Markdown, **`test` window, K2's verdict** |
| `sensitivity_test.csv` | `rakshak.eval.verdict` | CSV, same columns as `sensitivity.csv` |
| `figures/sensitivity_test.png` | `rakshak.eval.verdict` | PNG |
| `ablations.md` | `rakshak.eval.ablations` | Markdown, FR-018 |
| `lag_probe.md` | `rakshak.eval.lag_probe` | Markdown |
| `baf_validation.md` | `rakshak.eval.baf` | Markdown. **Optional step** — a clean checkout without the git-ignored 558 MB download still completes `make eval`. |
| `calibration_gap.md`, `calibration_profile.json` | `rakshak.data.profile` | Markdown + JSON. Not in the `make eval` chain. |

**The markdown artifacts have no schema beyond their headings.** They are built by `render_*`
functions that append to a `list[str]` line by line, and are **byte-identical for a fixed seed**
(NFR-003) — no wall-clock time, no host detail, no iteration order that depends on anything but
insertion order. A consumer that parses them is parsing a rendering and will break when the prose
changes. That is a real weakness of this boundary, and it is why T-0014's Build list opens with
`docs/CONTRACTS.md` — a versioned schema plus a shape test — as its Gate-0 precondition.
**`docs/CONTRACTS.md` does not exist yet**; when it does, it is authoritative for artifact schemas
and this section defers to it.

**The two CSVs are the only stable machine-readable surface.** Columns, one row per
(asymmetry point, model):

```
asymmetry, fp_cost_scale, model, savings, n_reviewed, n_held, hours_used,
binding_constraint, hold_threshold_median, hold_threshold_median_at_risk,
knapsack_ceiling, hindsight_ceiling, knapsack_clears_hold_everything,
margin_abs, margin_rel
```

`savings`, `margin_rel`, `asymmetry` and the two ceilings are dimensionless; `hours_used` is
hours; `binding_constraint` carries the same `"capacity"` / `"none"` vocabulary as `PolicyResult`.

**`results/reasons.json` (FR-014, T-0013) does not exist.** T-0014 names it as the source for the
mechanism view. `src/rakshak/explain/` currently contains only `__init__.py`. Until T-0013 lands,
any consumer must treat it as missing and say so on the page rather than substitute anything.

**`results/blackswan.md` (T-0022c), when it lands, is not part of the regenerable core.** The
black-swan stress test is a manually-invoked side track and is deliberately **outside** the
NFR-004 15-minute `make eval` budget; it is not chained into the `eval` target. Anyone auditing
"every number regenerable by `make eval`" should not expect to find it there, and the file itself
must say so.

---

## Known gaps at the time of writing

Recorded, not resolved — per `CLAUDE.md`'s working agreement that a spec error is raised rather
than papered over.

1. **Scores are not calibrated posteriors, but BMR treats them as such.** §1. T-0008 was cut.
2. **The markdown artifacts have no versioned schema.** §5. Owned by T-0014's `docs/CONTRACTS.md`.
3. **The dataset seam is module-level state**, not an argument on `Scorer` — neither thread-safe
   nor nestable across concurrent runs. §1, and the `ponytail:` comment in `splits.py`. Upgrade
   path is threading the dataset through `Scorer` and `evaluate_model`.
4. **`results/reasons.json` is cited by T-0014 and does not exist.** §5.
