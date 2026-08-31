"""The external anchor: Feedzai's Bank Account Fraud (BAF, NeurIPS 2022).

BAF is the **only** real-data reference this project has. CLAUDE.md is explicit that
there is no Razorpay data, so every claim about the generator being realistic rests on
the two gates that compare against BAF: G1 (marginal parity) and G2 (baseline transfer).
That is a lot of weight on one dataset, and this module's job is to make the dependency
explicit rather than incidental.

**BAF is not vendored.** It is ~1M rows under a research licence, it is not in the repo,
and ``make gates`` must run without it on a clean clone — so every function here returns
``None`` when the dataset is absent, and the gates report ``SKIP`` with the reason rather
than failing or, worse, silently passing.

To enable G1 and G2, put ``Base.csv`` (or a parquet of it) at ``data/external/baf/`` or
point ``RAKSHAK_BAF_PATH`` at the file.

**Caveat that belongs in the report, not in a comment nobody reads:** BAF is *account
opening* fraud, one row per application. Rakshak is *post-onboarding merchant* behaviour,
many rows per merchant per day. There is no row-level correspondence between them, and
pretending otherwise would be the kind of comparison that looks rigorous and means
nothing. The analogues below are therefore matched at the level of *distributional
family* — a velocity, a monetary magnitude, a device-reuse count, a cross-border flag —
and G1 asks whether our marginals are the same *shape*, not whether they are the same
quantity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "BAF_ENV_VAR",
    "BAF_SEARCH_DIR",
    "FeatureAnalogue",
    "ANALOGUES",
    "baf_path",
    "ks_statistic",
    "load_baf",
]

BAF_ENV_VAR = "RAKSHAK_BAF_PATH"
BAF_SEARCH_DIR = Path("data/external/baf")
_CANDIDATES = ("Base.parquet", "Base.csv", "baf.parquet", "baf.csv")


@dataclass(frozen=True, slots=True)
class FeatureAnalogue:
    """One BAF column paired with the generator observable of the same family.

    ``rakshak`` names a per-merchant-day aggregate the gate computes from the generated
    transaction stream. Both sides are rank-normalised before the KS test, because the
    units genuinely do not correspond — see the module docstring.
    """

    name: str
    baf_column: str
    rakshak: str
    why: str


#: Deliberately short. Four honest analogues beat twenty strained ones, and every extra
#: pairing is another place for a reviewer to ask what a "session length" has to do with
#: a kirana's Tuesday.
ANALOGUES: tuple[FeatureAnalogue, ...] = (
    FeatureAnalogue(
        name="velocity",
        baf_column="velocity_4w",
        rakshak="txn_count",
        why="both are a count of activity per entity per unit time; the heavy right tail "
        "is the property G1 is checking, not the absolute rate",
    ),
    FeatureAnalogue(
        name="monetary_magnitude",
        baf_column="proposed_credit_limit",
        rakshak="amount_inr",
        why="a right-skewed monetary magnitude attached to an entity. Lognormal-vs-"
        "Gaussian is exactly what makes t_wasserstein_7d trivial or not",
    ),
    FeatureAnalogue(
        name="device_reuse",
        baf_column="device_distinct_emails_8w",
        rakshak="payers_per_device",
        why="distinct identities behind one device — the same quantity g_device_reuse_rate "
        "measures, under a different name for identity",
    ),
    FeatureAnalogue(
        name="cross_border",
        baf_column="foreign_request",
        rakshak="is_international",
        why="a cross-border binary; matched on base rate rather than on shape",
    ),
)


def baf_path() -> Path | None:
    """Where BAF is, or ``None``. Checks ``RAKSHAK_BAF_PATH`` then the search directory."""
    override = os.environ.get(BAF_ENV_VAR)
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    for name in _CANDIDATES:
        candidate = BAF_SEARCH_DIR / name
        if candidate.exists():
            return candidate
    return None


def load_baf(columns: list[str] | None = None) -> pl.DataFrame | None:
    """Load BAF, or ``None`` if it is not present on this machine.

    Never raises for absence. A gate that cannot find its anchor must say so and record
    the fact; a gate that crashes on a clean clone breaks ``make all``, which charter K-5
    makes a stop-work condition.
    """
    path = baf_path()
    if path is None:
        return None
    if path.suffix == ".parquet":
        frame = pl.read_parquet(path)
    else:
        frame = pl.read_csv(path)
    if columns is not None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} does not look like BAF: missing column(s) {missing}. Expected the "
                f"NeurIPS 2022 Bank Account Fraud schema."
            )
        frame = frame.select(columns)
    return frame


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
