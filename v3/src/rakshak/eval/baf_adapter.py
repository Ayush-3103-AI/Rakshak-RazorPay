"""The external anchor: Feedzai's Bank Account Fraud (BAF, NeurIPS 2022).

BAF is the **only** external reference this project has, so G1b and G2 carry the entire
weight of the claim that the generator is not fiction. This module's job is to make that
dependency explicit, and — more importantly — to make it *honest*, because the honest
version of the comparison is much narrower than the spec assumed.

**Read this before trusting any number G1b or G2 prints.**

*BAF is bank account-opening applications.* One row per application. No amount, no
timestamp, no payer, no merchant, no sequences. ``data/external/baf.manifest.json`` says
so in the project's own words, and v1's ADR-0007 concluded it informs **none** of the
generator's marginals. Rakshak is post-onboarding merchant behaviour, many rows per
merchant per day. The shared feature space that ``08-generator-v2-spec.md`` §7 assumes
for G1 and G2 **largely does not exist**, and inventing one would produce a comparison
that looks rigorous and measures nothing.

*BAF is itself synthetic.* Jesus et al. generate it with a CTGAN fitted to a real
anonymised application dataset, under differential privacy. It is a *real label
distribution with real temporal drift* — which is genuinely valuable and is what T-0012
used it for — but it is not raw observation, and the fingerprints show: ``velocity_6h``
takes negative values, and ``velocity_*`` are non-integer despite being described as
counts of applications.

So the anchor is narrowed here to what survives scrutiny:

* **Three analogues that genuinely correspond** — two counts-of-events-at-an-entity-over-
  a-window, and one cross-border binary. These are scored.
* **Four columns named as NOT ANCHORABLE**, with the reason. They are recorded rather
  than dropped, because a quietly shortened list is how a weak anchor comes to look
  strong.

**BAF is not vendored and must not be.** Licence CC BY-NC-SA 4.0: non-commercial and
share-alike. It is not in the repo, it is not in git, and ``make gates`` must pass on a
clean clone without it — so every function here returns ``None`` when the dataset is
absent and the gates record ``SKIP`` with the reason. Point ``RAKSHAK_BAF_PATH`` at
``baf.zip`` (or an extracted ``Base.csv``/``Base.parquet``) to enable them.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "ANALOGUES",
    "BAF_COUNT_COLUMNS",
    "BAF_ENV_VAR",
    "BAF_SEARCH_DIR",
    "BAF_ZIP_MEMBER",
    "FeatureAnalogue",
    "MIN_DISTINCT_FOR_DISPERSION",
    "baf_path",
    "fano",
    "ks_statistic",
    "load_baf",
    "robust_standardise",
]

BAF_ENV_VAR = "RAKSHAK_BAF_PATH"
BAF_SEARCH_DIR = Path("data/external/baf")
BAF_ZIP_MEMBER = "Base.csv"
_CANDIDATES = ("Base.parquet", "Base.csv", "baf.parquet", "baf.csv", "baf.zip", "Base.zip")

#: Every column any gate reads. Loaded once as a projection: ``Base.csv`` is 213 MB and
#: 32 columns wide, and none of the other 25 columns has a Rakshak counterpart.
_COLUMNS = (
    "fraud_bool",
    "month",
    "zip_count_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "device_distinct_emails_8w",
    "foreign_request",
    # Not analogues. Loaded so that (a) every NOT-ANCHORABLE claim in ``ANALOGUES``
    # below is checkable against the data rather than taken on trust, and (b) G2 can
    # report how much signal BAF's wider numeric schema carries, which is the number
    # that shows the three-column shared subspace is near-empty. Numeric only: BAF's
    # five string columns would need an encoding, and none of them has a counterpart.
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "session_length_in_minutes",
    "proposed_credit_limit",
    "credit_risk_score",
    "name_email_similarity",
    "days_since_request",
    "customer_age",
    "income",
)

#: BAF columns that are genuine integer counts of events over a stated window, and so
#: have a Fano factor that means something. ``velocity_6h/24h/4w`` are deliberately
#: **excluded**: they are non-integer, ``velocity_6h`` goes negative, and a Fano factor is
#: not scale-free, so quoting one for a rate in arbitrary units is quoting the units.
BAF_COUNT_COLUMNS = (
    "zip_count_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "device_distinct_emails_8w",
)

#: A column with fewer distinct values than this has no dispersion left to measure and is
#: reported but not gated on. Stated here, ahead of the filter being applied, so that
#: excluding ``device_distinct_emails_8w`` (four values, 96.8% of them exactly 1) is a
#: declared rule rather than a convenient one.
MIN_DISTINCT_FOR_DISPERSION = 10


@dataclass(frozen=True, slots=True)
class FeatureAnalogue:
    """One BAF column paired with the generator observable of the same family.

    ``anchorable=False`` records a pairing the spec implies but the data does not
    support. Those are printed by G1b with their reason and never scored — naming them is
    the point, since the alternative is a list that has been quietly shortened until it
    passes.
    """

    name: str
    baf_column: str
    rakshak: str
    anchorable: bool
    why: str


#: Three scored, four refused. ``rakshak`` names an aggregate G1b computes from the
#: generated stream; the pairing rule is *count of events at one entity over one window*
#: on both sides, matched on window length, because that is the only structural
#: correspondence between an application funnel and a payments funnel that survives being
#: asked about.
ANALOGUES: tuple[FeatureAnalogue, ...] = (
    FeatureAnalogue(
        name="count_28d",
        baf_column="zip_count_4w",
        rakshak="txn_per_merchant_28d",
        anchorable=True,
        why="applications at one zip code over 4 weeks against transactions at one "
        "merchant over 28 days. Both are integer counts of events accruing to a single "
        "entity over a matched window, and the window is matched deliberately: a Fano "
        "factor and a KS both move with window length, so an unmatched comparison would "
        "be measuring the window",
    ),
    FeatureAnalogue(
        name="count_56d",
        baf_column="bank_branch_count_8w",
        rakshak="txn_per_merchant_56d",
        anchorable=True,
        why="the same pairing at BAF's other window length — applications at one bank "
        "branch over 8 weeks against transactions at one merchant over 56 days. Included "
        "even though it is the analogue that comes out worst, because dropping it would "
        "leave the gate resting on the single window that happened to agree",
    ),
    FeatureAnalogue(
        name="cross_border",
        baf_column="foreign_request",
        rakshak="is_international",
        anchorable=True,
        why="a cross-border binary on both sides. For a binary the two-sample KS "
        "statistic is exactly the base-rate difference, so the spec's KS <= 0.15 needs no "
        "reinterpretation here — it is the same test, applied to the only quantity a "
        "binary has",
    ),
    FeatureAnalogue(
        name="monetary_magnitude",
        baf_column="proposed_credit_limit",
        rakshak="amount_inr",
        anchorable=False,
        why="NOT ANCHORABLE. `proposed_credit_limit` takes 12 distinct values on a bank's "
        "offer grid (190, 200, 210, 490, ... 2100) and is a limit the bank offered, not "
        "money that moved. `amount_inr` is a realised continuous payment spanning four "
        "orders of magnitude. There is no shape to compare: one side is a menu",
    ),
    FeatureAnalogue(
        name="device_reuse",
        baf_column="device_distinct_emails_8w",
        rakshak="payers_per_device",
        anchorable=False,
        why="NOT ANCHORABLE. The column has four distinct values (-1, 0, 1, 2) and 96.8% "
        "of the million rows are exactly 1; -1 is a missing-value sentinel, not a count. "
        "Its Fano factor is 0.032 — underdispersed almost to a constant. A KS against it "
        "would be measuring a censoring rule at Feedzai, not device reuse",
    ),
    FeatureAnalogue(
        name="velocity",
        baf_column="velocity_4w",
        rakshak="txn_count",
        anchorable=False,
        why="NOT ANCHORABLE, and this is the pairing that looked most obviously right. "
        "`velocity_4w` is not a count: it is non-integer, `velocity_6h` goes negative, "
        "and it is near-symmetric (skew -0.06) with a coefficient of variation of 0.19. "
        "A merchant's daily transaction count is a heavy-tailed overdispersed count "
        "(skew ~16). Same word, different family",
    ),
    FeatureAnalogue(
        name="session_length",
        baf_column="session_length_in_minutes",
        rakshak="(none)",
        anchorable=False,
        why="NOT ANCHORABLE. Rakshak has no session. There is no generator observable "
        "this could be paired with, and pairing it with one anyway is the failure mode "
        "this list exists to prevent",
    ),
)


def baf_path() -> Path | None:
    """Where BAF is, or ``None``. Checks ``RAKSHAK_BAF_PATH`` then the search directory.

    An override that points at nothing returns ``None`` rather than falling back to the
    search directory. That is what makes the clean-clone SKIP path testable on a machine
    that does have the dataset: point the variable at a path that does not exist.
    """
    override = os.environ.get(BAF_ENV_VAR)
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    for name in _CANDIDATES:
        candidate = BAF_SEARCH_DIR / name
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _read(path: Path) -> pl.DataFrame:
    """Read ``_COLUMNS`` out of BAF once per process.

    Projected, never eager over all 32 columns: the zip route reads and parses only the
    seven columns any gate touches, which is ~1.5 s against ~213 MB of CSV. Cached because
    G1b and G2 both want it and a second parse buys nothing.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive, archive.open(BAF_ZIP_MEMBER) as handle:
            return pl.read_csv(handle, columns=list(_COLUMNS))
    if path.suffix == ".parquet":
        return pl.read_parquet(path, columns=list(_COLUMNS))
    return pl.read_csv(path, columns=list(_COLUMNS))


def load_baf(columns: list[str] | None = None) -> pl.DataFrame | None:
    """Load BAF, or ``None`` if it is not present on this machine.

    Never raises for absence. A gate that cannot find its anchor must say so and record
    the fact; a gate that crashes on a clean clone breaks ``make all``, which charter K-5
    makes a stop-work condition.
    """
    path = baf_path()
    if path is None:
        return None
    frame = _read(path)
    if columns is None:
        return frame
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} does not look like BAF: missing column(s) {missing}. Expected the "
            f"NeurIPS 2022 Bank Account Fraud schema."
        )
    return frame.select(columns)


def fano(values: npt.NDArray[np.float64]) -> float:
    """Variance over mean. ``1.0`` for a Poisson process; ``> 1`` is overdispersion.

    Only meaningful for a **count**. Fano is not scale-free — multiplying a variable by
    ``c`` multiplies its Fano by ``c`` — so applying it to a rate in arbitrary units
    reports the units and nothing else. See ``BAF_COUNT_COLUMNS``.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    mean = float(array.mean()) if array.size else float("nan")
    return float(array.var() / mean) if mean else float("nan")


def robust_standardise(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Centre on the median, scale by the IQR. Removes units, keeps shape.

    This replaces the rank normalisation G1b used before T-116b, which was **vacuous**:
    mapping a sample to its own ranks makes its empirical CDF uniform *by construction*,
    so the KS between two rank-normalised samples is ~0 whatever they are. Measured, at a
    million draws each: ``KS(rank(Normal), rank(Exponential)) == 0.0`` exactly. G1b was
    guaranteed to pass and could not have detected anything.

    Median and IQR rather than mean and standard deviation because both sides are
    heavy-tailed counts, where a handful of extreme values would otherwise set the scale
    and flatten everything else into the first bin. A zero IQR (a binary, a near-constant
    column) leaves the scale at 1, which is correct: the KS between two 0/1 samples is
    then exactly the base-rate difference.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return array
    q1, q2, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    spread = float(q3 - q1)
    return (array - float(q2)) / (spread if spread > 0.0 else 1.0)


def ks_statistic(
    left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]
) -> float:
    """Two-sample Kolmogorov-Smirnov statistic, computed without scipy's p-value.

    Only the statistic is wanted: G1's GREEN condition is ``KS <= 0.15``, and at BAF's
    ~1M rows against our ~15M every p-value is zero regardless of whether the two
    distributions differ in any way a human would care about. Reporting a p-value here
    would be reporting the sample size.
    """
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    if left.size == 0 or right.size == 0:
        return float("nan")
    grid = np.concatenate([left, right])
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size
    return float(np.max(np.abs(cdf_left - cdf_right)))
