"""Global constants: seeds, paths, cost defaults.

Single source of truth. Nothing in this repo hard-codes a seed, a path, or a
cost parameter anywhere else.

Every cost primitive below carries a **source class**, a citation or an explicit
`ASSUMPTION` tag, and a range, copied from `07-math.md` §5 (as amended
2026-08-28 by T-0017, implemented by T-0007a):

* **[S]** sourced — a public citation is given.
* **[D]** derived — computed from other rows here.
* **[A]** `ASSUMPTION` — no public source found; range stated.

The sourced/assumed distinction is deliberately visible per constant rather than
averaged away: six primitives are still assumptions, `MERCHANT_LIFETIME_MONTHS`
is the weakest, and FR-020's sensitivity sweep over `COST_PRIMITIVE_RANGES` is
what makes the headline claim defensible, not the central values. The README and
the video must say so.

**No value here was chosen by the metric it produces.** `07-math.md` §5 demotes
the 400-600 FP-to-loss asymmetry from a gate to a reported cross-check; a
primitive changes only when its *source* changes, and the change is recorded in
`LOGBOOK.md`.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

SEED: int = 42
"""Global seed. Every script takes --seed and defaults to this (CLAUDE.md)."""

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------
# Cost matrix (07-math.md §5, amended 2026-08-28 by T-0017, implemented T-0007a).
# Units: INR unless stated. Source class, citation/ASSUMPTION and range on every
# constant; machine-readable ranges in COST_PRIMITIVE_RANGES at the bottom.
# --------------------------------------------------------------------------

TAU_REVIEW_HOURS: float = 0.067
"""tau — analyst hours consumed by one manual review (~4 min).
Range 0.05 - 0.12 h. Class **[S]**: Razorpay Engineering, Dec 2025 — stated
per-review time. Cross-checks against the 700-800 analyst-hours/month figure in
`00-charter.md` §1."""

WAGE_ANALYST_INR_PER_HOUR: float = 600.0
"""w_analyst — fully-loaded Indian risk-ops analyst cost. Units: INR / hour.
Range 300 - 700. Class **[S+A]**: PayScale India *Fraud Analyst* INR 4.19 L/yr and
Glassdoor India INR 4.48 L/yr (2026) give INR 210-225/h at 2,000 h/yr [S]; the
1.5-1.8x fully-loaded multiplier (benefits, seat, QA, supervision) is an
**ASSUMPTION**. Held deliberately at the upper end of the band: an expensive
analyst makes REVIEW look costly, which is conservative *against* Rakshak's own
capacity story."""

COST_REVIEW_INR: float = TAU_REVIEW_HOURS * WAGE_ANALYST_INR_PER_HOUR
"""c_rev = tau * w_analyst — cost of one REVIEW action. ~INR 40.
Range 15 - 84. Class **[D]**."""

P_CHURN_GIVEN_HOLD: float = 0.35
"""P(merchant churns | wrongly held). Dimensionless.
Range 0.15 - 0.60. Class **[A] ASSUMPTION**: informed by the public-review
pattern — frozen settlements are the most common Razorpay complaint
(`00-charter.md` §1). A pattern, not a measurement."""

COST_SUPPORT_INR: float = 500.0
"""c_support — escalation handling cost per HOLD. Units: INR.
Range 200 - 1,500. Class **[A] ASSUMPTION**, loosely bounded above by published
dispute-handling fees: Visa VAMP's excessive tier charges US$8 (~INR 700) per
dispute from 1 Apr 2026."""

P_ANALYST_MISS: float = 0.15
"""p_miss — probability a review clears a merchant who is truly bad.
Range 0.05 - 0.30. Class **[A] ASSUMPTION**: no public source."""

RESIDUAL_LEAKAGE_RHO: float = 0.10
"""rho — fraction of loss still leaking between the HOLD decision and settlement
actually stopping. Dimensionless.
Range 0.05 - 0.25. Class **[A] ASSUMPTION**: no public source."""

GROSS_MARGIN_RATE: float = 0.0010
"""g — platform gross margin per rupee of processed volume (10 bps of TPV).
Dimensionless. Range 0.0008 - 0.0015. Class **[S]**.

**This replaces `MDR_RATE = 0.02`, which was a price, not a margin.** 2% is what
the *merchant pays*; almost all of it leaves again as issuer interchange, scheme
fees and GST. Razorpay FY24: revenue ~INR 2,501 Cr against annualised TPV
~US$180 bn gives a take rate of ~0.27% of TPV; gross profit INR 906 Cr FY24
(INR 1,277 Cr FY25) gives a gross margin of ~36% of revenue. g ~ 0.0027 * 0.36.
Secondary sources reporting company disclosures, not an audited filing — flagged
as such (07-math.md §5, citation 5)."""

MERCHANT_LIFETIME_MONTHS: float = 30.0
"""l_m — expected *remaining* merchant lifetime on the platform. Units: months.
Range 18 - 48. Class **[A] ASSUMPTION**.

**No public disclosure of Indian payment-aggregator merchant retention exists.**
The range brackets a 2.1-5.6% monthly churn rate. This is the least-defensible
number in the cost model and FR-020 must sweep it."""

CHARGEBACK_REALISATION_RATE: float = 0.05
"""r_cb — fraction of bad-state turnover that returns as chargeback,
confirmed-fraud write-off or unrecovered negative balance. Dimensionless.
Range 0.02 - 0.20. Class **[A] ASSUMPTION, bracketed by [S] anchors**.

Floor anchors, both cited: Nilson puts *all-merchant* card fraud at 6.43c per
US$100 (0.064% of volume, 2024) — a population floor far below any individual bad
merchant; card-scheme monitoring programmes define where a merchant becomes
formally abnormal (Mastercard ECM 1.5%, HECM 3.0%; Visa VAMP "excessive" 1.5%
from 1 Apr 2026). A merchant Rakshak exists to catch is by construction above
those, so the range starts at 2%. Ceiling anchor: a terminal bust-out window can
approach total dispute, hence 20%. The central 0.05 sits just above the
scheme-excessive boundary and **was chosen from these anchors before the
resulting FP-to-loss ratio was computed** (07-math.md §5, citations 1-2)."""

ANCILLARY_LOADING_PHI: float = 0.35
"""phi — ancillary loading on realised loss: scheme dispute fees, representment
handling, monitoring-programme penalties. Dimensionless.
Range 0.20 - 0.50. Class **[A] ASSUMPTION, bracketed by [S]**.

LexisNexis *True Cost of Fraud* puts total cost at 3.84x face value in India
(2021) and 3.95x in APAC (2023) — but that multiplier includes internal labour and
recovery effort, which this cost matrix already charges separately via c_rev and
c_support. Double-counting would be an error, so phi covers scheme/dispute
ancillaries only and the LexisNexis figure is a **hard upper bound, not the
value** (07-math.md §5, citation 3)."""

REVIEW_CAPACITY_HOURS: float = 40.0
"""K — absolute analyst-hours available per decision period. Kept for callers that want
a fixed pool; the eval harness does NOT use it (see ADR-0008). At tau = 0.067 h this buys
597 reviews, which is slack against any split of <= 300 merchants, so as an evaluation
budget it makes the capacity constraint do nothing."""

REVIEW_CAPACITY_HOURS_PER_1000_MERCHANTS: float = 4.0
"""Review capacity expressed per 1000 merchants under watch, so it scales with the
population being scored (ADR-0008). At TAU_REVIEW_HOURS = 0.067 this is ~60 reviews per
1000 merchants (~6% of the book per decision period) — a plausible risk-ops load. The
harness derives K from this and the size of the split it is scoring; FR-017's constraint
only binds when it is expressed this way."""

# --------------------------------------------------------------------------
# Model defaults
# --------------------------------------------------------------------------

N_HIDDEN_STATES: int = 4
"""K for the HMM. Provisional; settled empirically by BIC sweep in T-0004."""

ARI_RECOVERY_THRESHOLD: float = 0.5
"""FR-013 gate: recovered-vs-true state ARI must exceed this."""

# --------------------------------------------------------------------------
# Synthetic generator (T-0003). Data is git-ignored; regenerate with
# `python -m rakshak.generator.generate --seed 42`.
# --------------------------------------------------------------------------

SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"
TRANSACTIONS_PARQUET: Path = SYNTHETIC_DIR / "transactions.parquet"
STATE_PATHS_PARQUET: Path = SYNTHETIC_DIR / "state_paths.parquet"

SYNTHETIC_SHOCK_DIR: Path = DATA_DIR / "synthetic_shock"
"""Black-swan stress-test dataset (T-0022a). A SEPARATE population carrying a
population-wide shock on chosen days, written by
`python -m rakshak.generator.generate --shock-day D --shock-magnitude M`.

It is scored with the same `SPLIT_DAY_BOUNDS` / `MERCHANT_GROUP_CYCLE` geometry as
`SYNTHETIC_DIR` — a new dataset, not a new split design. Nothing in the frozen
results (`verdict.md`, `ablations.md`, `lag_probe.md`, `baf_validation.md`) reads it,
and the generator refuses to write shocked data into `SYNTHETIC_DIR`."""

GENERATOR_START_DATE: str = "2026-01-01"
"""ISO date of horizon day 0. A Thursday — the weekday seasonality table assumes it."""

N_MERCHANTS: int = 500
"""Default merchant population size."""

HORIZON_DAYS: int = 270
"""Default observation horizon in days (9 months; test window is months 8-9)."""

FRAUD_MERCHANT_RATE: float = 0.20
"""Fraction of merchants assigned a typology. Far above real prevalence, chosen so each of
the 5 typologies has enough merchants for a credible per-class metric. Must be stated in the
README next to any per-typology number."""

# --------------------------------------------------------------------------
# Feature layer (T-0004)
# --------------------------------------------------------------------------

WINDOW_DAYS: int = 7
"""Length of one emission window in days. 270-day horizon / 7 = 39 windows per merchant.
08-pseudocode.md §C names 7 days as the default."""

BURN_IN_WINDOWS: int = 8
"""Leading windows used to estimate each merchant's own location and scale (FR-007).
8 windows = 56 days. Chosen so the burn-in ends strictly before the earliest possible
typology onset (`generator.MIN_ONSET_DAY` = 63 days = 9 windows). Every merchant is
therefore verifiably HEALTHY throughout its own burn-in, which is asserted in
tests/test_hmm_recovery_fullscale.py::test_emission_panel_shape_and_alignment."""

SHRINKAGE_N0: float = 30.0
"""n_0 in the empirical-Bayes weight w_m = n_m / (n_m + n_0), 07-math.md §3. Units are
burn-in TRANSACTIONS, not windows — see the deviation note in features/standardise.py.
Governs SCALE shrinkage only; location is never shrunk (also in that note)."""

MIN_SEGMENT_MERCHANTS: int = 20
"""FR-011's floor: every training segment holds at least this many merchants."""

MAX_AOV_BANDS: int = 3
"""Cap on AOV bands per MCC, so segment labels stay readable as LOW / MID / HIGH."""

STANDARDISE_EPS: float = 1e-6
"""Denominator guard in the z-score, and the threshold below which a scale estimate is
treated as degenerate and the next rung of the merchant -> segment -> population cascade
is used instead."""

Z_CLIP: float = 10.0
"""Standardised emissions are winsorised to +/- this. A numerical guard, applied
symmetrically to every feature before any metric is computed, not a tuning knob: several
features are zero-inflated, so a merchant whose burn-in never saw the event has a
near-zero own-scale and a single later event would otherwise land at hundreds of sigma
and dominate every Gaussian emission. FR-013's separation gate lives near 1 sigma, so
nothing that matters is clipped."""

VULCAN_SCORE_COLUMN: str = "risk_score"
"""Optional per-transaction risk-score column (FR-010). When present, its window mean and
p95 enter the emission vector; when absent the pipeline runs without it and logs so."""

# --------------------------------------------------------------------------
# Frozen evaluation split (06-requirements.md §3, T-0005)
# --------------------------------------------------------------------------

DAYS_PER_MONTH: int = 30
"""Generator months are exactly 30 days; 9 * 30 == HORIZON_DAYS."""

SPLIT_DAY_BOUNDS: dict[str, tuple[int, int]] = {
    "train": (0, 6 * DAYS_PER_MONTH),
    "validate": (6 * DAYS_PER_MONTH, 7 * DAYS_PER_MONTH),
    "test": (7 * DAYS_PER_MONTH, 9 * DAYS_PER_MONTH),
}
"""Temporal windows [start_day, end_day) in days since GENERATOR_START_DATE.
Frozen: train months 1-6, validate month 7, test months 8-9. Changing this is a
DESCEND, not an edit — `eval.splits.assert_window_is_frozen` fails the build."""

MERCHANT_GROUP_CYCLE: tuple[str, ...] = ("train", "train", "train", "validate", "test")
"""Deterministic 3:1:1 interleave over sorted merchant IDs within each typology.
Gives a 60/20/20 merchant-group split, stratified so every split sees all five
typologies. Seed-independent on purpose: the frozen eval must not move when
someone changes --seed. The generator reads this too, so that a merchant's typology
onset lands inside the same window its merchant group is scored on (ADR-0008's
sibling finding, T-0003b)."""

BAD_STATES: frozenset[str] = frozenset({"RAMP", "FRAUD", "DORMANT"})
"""Ground-truth "bad" latent states. RAMP is deliberately included: SLOW_RAMP
merchants enter RAMP and never reach FRAUD, so excluding it would make the
adversarial typology undetectable by construction rather than by difficulty
(T-0003 finding). DORMANT is included because this generator only emits it as
the bust-out aftermath."""

# --------------------------------------------------------------------------
# Cost-primitive uncertainty ranges (07-math.md §5) — machine-readable, so
# FR-020's sensitivity sweep reads the same numbers the docstrings state.
# --------------------------------------------------------------------------

COST_PRIMITIVE_RANGES: dict[str, tuple[float, float]] = {
    "TAU_REVIEW_HOURS": (0.05, 0.12),
    "WAGE_ANALYST_INR_PER_HOUR": (300.0, 700.0),
    "COST_REVIEW_INR": (15.0, 84.0),
    "P_CHURN_GIVEN_HOLD": (0.15, 0.60),
    "COST_SUPPORT_INR": (200.0, 1500.0),
    "P_ANALYST_MISS": (0.05, 0.30),
    "RESIDUAL_LEAKAGE_RHO": (0.05, 0.25),
    "GROSS_MARGIN_RATE": (0.0008, 0.0015),
    "MERCHANT_LIFETIME_MONTHS": (18.0, 48.0),
    "CHARGEBACK_REALISATION_RATE": (0.02, 0.20),
    "ANCILLARY_LOADING_PHI": (0.20, 0.50),
}
"""(low, high) plausible range per cost primitive, keyed by constant name.

Copied from `07-math.md` §5's primitive table. `tests/test_cost.py` asserts every
shipping central value lies inside its own range, so the two cannot drift apart.

`MDR_RATE = 0.02` was **deleted** here at T-0007a: it was the merchant-facing
price standing in for the platform's gross margin, and overstated V_m ~20x. Use
`GROSS_MARGIN_RATE`."""
