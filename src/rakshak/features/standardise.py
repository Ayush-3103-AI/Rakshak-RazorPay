"""Within-merchant standardisation (FR-007) and segment assignment (FR-011).

This is P-02, the single most important modelling decision in the repo. Every emission
is expressed as a deviation from *that merchant's own* burn-in norm, never from a
population norm. It is what lets one pooled HMM span a 60-INR food-delivery merchant and
a 30 000-INR jeweller without declaring the jeweller anomalous for being a jeweller —
the classic 2008-era cardholder-HMM false-positive failure.

Maths: `07-math.md` §3. Pseudocode: `08-pseudocode.md` §C.

    mu_m = mean(burn_m)                                    # see the deviation note below
    sd_m = w_m * std(burn_m) + (1 - w_m) * std(burn_segment)
    w_m  = n_m / (n_m + n0)
    z    = (x - mu_m) / (sd_m + eps)

DEVIATION FROM 07-math.md §3 — RAISED, NOT PATCHED AROUND. Needs ratification.
------------------------------------------------------------------------------
The spec shrinks BOTH location and scale toward the segment. Location shrinkage
contradicts FR-007's own acceptance test, demonstrably and unavoidably:

    Two merchants with identical relative behaviour and 100x different AOV must produce
    near-identical standardised emissions. Under the spec's formula they do not, unless
    w_m = 1 exactly. Measured on the pair the test constructs: max gap 1.73 sigma on
    `log_amount_mean` even with a healthy w_m = 0.89 and a correct MCC x AOV-band
    segmentation that puts the two in different bands. With location shrinkage disabled
    the same pair matches to 9e-5, i.e. exactly, up to 2-decimal rounding of INR amounts.

The mechanism: the between-merchant spread of `log_amount_mean` inside one AOV band is
about 1.0 log unit, while a single merchant's window-to-window spread of the same
quantity is about 0.07. Shrinking the location by even 11% therefore moves the merchant
roughly 1.5 of its OWN standard deviations away from zero, purely for being larger than
its band's average. That is precisely the "flag the jeweller for being a jeweller"
false-positive mode FR-007 exists to prevent, reintroduced through the shrinkage term.

Resolution implemented here, pending sign-off: shrink the SCALE toward the segment, take
the LOCATION from the merchant's own burn-in always. Scale is what a thin history
genuinely cannot estimate; location is what carries the merchant's identity, and
borrowing it is what breaks the invariant. This makes FR-007's acceptance test hold by
construction rather than by tolerance. `n0` now governs scale shrinkage only.

n_m is read as the burn-in TRANSACTION count, not window count. 07-math.md §3 says only
"observation count". Read as windows, every merchant has n_m = 8 and w_m = 0.21
regardless of history depth, which makes the weight carry no information about the thing
it is supposed to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rakshak.config import (
    BURN_IN_WINDOWS,
    MAX_AOV_BANDS,
    MIN_SEGMENT_MERCHANTS,
    SHRINKAGE_N0,
    STANDARDISE_EPS,
    Z_CLIP,
)
from rakshak.features.windows import _MID, _WIN

_BAND_NAMES: tuple[str, ...] = ("LOW", "MID", "HIGH")


@dataclass(frozen=True)
class SegmentMap:
    """MCC x AOV-band segment definition, fitted on a training population (FR-011).

    Holding the band edges rather than recomputing them per call is what lets held-out
    merchants be segmented with training-set information only, so segmentation cannot
    leak the test distribution.

    Attributes:
        edges: MCC -> ascending interior AOV cut points in INR. An MCC with one band
            maps to an empty tuple.
    """

    edges: dict[str, tuple[float, ...]]

    def assign(self, mcc: pd.Series, aov: pd.Series) -> pd.Series:
        """Map merchants to segment labels of the form ``"5411:MID"``.

        Args:
            mcc: Merchant category code per merchant, indexed by merchant_id.
            aov: Burn-in mean ticket size in INR per merchant, same index.

        Returns:
            String Series of segment labels, same index as ``mcc``.
        """
        out = pd.Series(index=mcc.index, dtype=object)
        for code, cuts in self.edges.items():
            mask = mcc == code
            if not mask.any():
                continue
            band = np.searchsorted(np.asarray(cuts, dtype=float), aov[mask].to_numpy(dtype=float))
            out.loc[mask] = [f"{code}:{_BAND_NAMES[b]}" for b in band]
        # An MCC unseen in training falls back to its own single-band segment rather
        # than silently borrowing another category's scale.
        unseen = out.isna()
        if unseen.any():
            out.loc[unseen] = mcc[unseen].astype(str) + ":LOW"
        return out.astype(str)


def fit_segment_map(
    mcc: pd.Series,
    aov: pd.Series,
    min_merchants: int = MIN_SEGMENT_MERCHANTS,
    max_bands: int = MAX_AOV_BANDS,
) -> SegmentMap:
    """Choose AOV band edges per MCC so that every segment holds >= ``min_merchants``.

    The band count for an MCC is ``clip(floor(n_mcc / min_merchants), 1, max_bands)`` and
    the cuts are equal-count quantiles, so the floor holds by construction rather than by
    hope. An MCC with fewer than ``2 * min_merchants`` merchants stays a single segment.

    Args:
        mcc: Merchant category code per merchant, indexed by merchant_id.
        aov: Burn-in mean ticket size in INR per merchant, same index.
        min_merchants: FR-011's population floor per segment.
        max_bands: Cap on bands per MCC; 3 keeps the labels readable (LOW/MID/HIGH).

    Returns:
        A fitted `SegmentMap`.
    """
    edges: dict[str, tuple[float, ...]] = {}
    for code, group in aov.groupby(mcc):
        n_bands = int(np.clip(len(group) // max(min_merchants, 1), 1, max_bands))
        if n_bands < 2:
            edges[str(code)] = ()
            continue
        quantiles = np.linspace(0.0, 1.0, n_bands + 1)[1:-1]
        edges[str(code)] = tuple(np.quantile(group.to_numpy(dtype=float), quantiles))
    return SegmentMap(edges=edges)


@dataclass(frozen=True)
class EmissionSet:
    """Standardised emissions for a merchant population, ready for `models.hmm.HMM`.

    Attributes:
        merchant_ids: Merchant ids in row order of ``X``, shape (M,).
        window_index: Absolute calendar window indices, shape (W,).
        X: Standardised emissions, shape (M, W, D), dimensionless z-scores.
        feature_names: Length-D feature order matching ``X``'s last axis.
        segments: Segment label per merchant, shape (M,).
        burn_in_windows: Number of leading windows used to fit location and scale.
        shrinkage_weight: w_m per merchant, shape (M,). 1.0 means no shrinkage.
        segment_map: The fitted segmentation, reusable on a held-out population.
        n_txn: Raw transaction count per merchant-window, shape (M, W), dimensionless
            counts. Carried unstandardised on purpose: it is the only exact,
            threshold-free way for a caller to ask "was this window empty", which is
            the deterministic DORMANT rule of T-0004b. The standardised `sparse`
            emission cannot answer that — a merchant whose burn-in held no empty window
            has a degenerate own-scale for it and lands on the Z_CLIP rail.
    """

    merchant_ids: np.ndarray
    window_index: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    segments: np.ndarray
    burn_in_windows: int
    shrinkage_weight: np.ndarray
    segment_map: SegmentMap
    n_txn: np.ndarray

    def sequences(self) -> list[np.ndarray]:
        """Return one (W, D) float64 observation array per merchant, in row order."""
        return [np.ascontiguousarray(self.X[i]) for i in range(self.X.shape[0])]


def standardise_panel(
    panel: pd.DataFrame,
    feature_names: tuple[str, ...],
    burn_in_windows: int = BURN_IN_WINDOWS,
    n0: float = SHRINKAGE_N0,
    segment_map: SegmentMap | None = None,
    standardise: bool = True,
) -> EmissionSet:
    """Standardise a window panel within merchant, shrinking to segment (FR-007).

    Args:
        panel: Dense panel from `windows.build_window_features`, indexed by
            (merchant_id, window_index), carrying `mcc` and `n_txn`.
        feature_names: Columns of ``panel`` that form the emission vector.
        burn_in_windows: Number of leading windows used to estimate location and scale.
            Must be >= 2 and strictly less than the panel's window count — the burn-in
            ends before the evaluation window, which is the leakage guard `eval/splits.py`
            enforces from the other side.
        n0: Scale-shrinkage constant, in burn-in transactions. 0 disables shrinkage and
            makes standardisation exactly scale-invariant (see the module docstring).
        segment_map: A segmentation fitted elsewhere, e.g. on the training merchants. When
            None, one is fitted on this population.
        standardise: When False the emissions are the raw panel values, in the units of
            `windows.FEATURE_UNITS` - no within-merchant location, no scale, no segment
            shrinkage and no Z_CLIP winsorising. That is FR-018's standardisation-off
            ablation and nothing else; the default True is FR-007's shipping path and is
            what every caller outside `eval/ablations.py` uses. The segment map is still
            fitted and returned when False, so the held-out contract is unchanged, but
            nothing consumes it.

    Returns:
        An `EmissionSet` whose ``X`` is dimensionless: 0 means "this merchant's own
        burn-in normal", 1 means "one of this merchant's own burn-in standard deviations
        away from it".

    Raises:
        ValueError: If ``burn_in_windows`` does not leave at least one later window.
    """
    merchants = panel.index.get_level_values(_MID).unique().to_numpy()
    windows = panel.index.get_level_values(_WIN).unique().to_numpy()
    n_merchants, n_windows, n_features = len(merchants), len(windows), len(feature_names)
    if not 2 <= burn_in_windows < n_windows:
        raise ValueError(
            f"burn_in_windows must satisfy 2 <= b < {n_windows}; got {burn_in_windows}"
        )

    raw = panel[list(feature_names)].to_numpy(dtype=float).reshape(
        n_merchants, n_windows, n_features
    )
    burn = raw[:, :burn_in_windows, :]
    mu_m = burn.mean(axis=1)
    sd_m = burn.std(axis=1)

    # n_m: burn-in transaction count, one per merchant.
    n_txn = panel["n_txn"].to_numpy(dtype=float).reshape(n_merchants, n_windows)
    n_m = n_txn[:, :burn_in_windows].sum(axis=1)

    # Segments (FR-011). AOV is the merchant's own burn-in mean ticket size in INR, so a
    # merchant is banded by what it did before anything could have gone wrong.
    log_amount_idx = feature_names.index("log_amount_mean")
    aov = pd.Series(np.exp(mu_m[:, log_amount_idx]), index=merchants)
    mcc = pd.Series(
        panel["mcc"].to_numpy().reshape(n_merchants, n_windows)[:, 0].astype(str), index=merchants
    )
    if segment_map is None:
        segment_map = fit_segment_map(mcc, aov)
    segments = segment_map.assign(mcc, aov).to_numpy()

    # Segment-level pooled burn-in scale, computed on the same burn-in windows so nothing
    # after the burn-in can influence the standardisation. The segment LOCATION is
    # deliberately not computed: see the module docstring's deviation note.
    seg_sd = np.empty_like(sd_m)
    for label in np.unique(segments):
        members = segments == label
        seg_sd[members] = burn[members].reshape(-1, n_features).std(axis=0)

    # 08-pseudocode.md §C failure note: a feature with no variance in a merchant's own
    # burn-in falls back entirely to the segment scale, rather than dividing by ~0.
    # Some features (`sparse`, `chargeback_ratio`) are constant across a whole SEGMENT's
    # burn-in too, so the cascade has to continue to the population before it gives up:
    # merchant scale -> segment scale -> population scale -> 1.0. Without the population
    # rung a genuinely rare event divides by eps and lands at 1e8 sigma, which then
    # dominates every Gaussian emission in the HMM.
    pop_sd = burn.reshape(-1, n_features).std(axis=0)
    pop_sd = np.where(pop_sd > STANDARDISE_EPS, pop_sd, 1.0)
    seg_sd = np.where(seg_sd > STANDARDISE_EPS, seg_sd, pop_sd[None, :])
    sd_m = np.where(sd_m > STANDARDISE_EPS, sd_m, seg_sd)

    # Location: the merchant's own burn-in, never shrunk. See the module docstring's
    # deviation note � shrinking it toward the segment breaks FR-007's own acceptance
    # test by up to 1.7 sigma and reintroduces the false-positive mode P-02 prevents.
    weight = (n_m / (n_m + n0))[:, None]
    mu = mu_m
    sd = weight * sd_m + (1.0 - weight) * seg_sd

    # Numerical guard, not a tuning knob: applied symmetrically to every feature before
    # any metric is computed. Several features are zero-inflated (chargeback_ratio,
    # chargeback_lag_days, sparse) so a merchant whose burn-in never saw the event has a
    # near-zero own-scale even after the cascade, and a single later event lands at
    # hundreds of sigma. A diagonal-covariance Gaussian HMM responds by inflating one
    # state's variance until that state absorbs everything. Winsorising at +/-10 sigma
    # costs nothing that matters: FR-013's separation gate lives at ~1 sigma.
    # FR-018 ablation branch: `standardise=False` returns `raw` untouched, which also
    # skips the Z_CLIP winsoriser - clipping raw INR-scale features at +/-10 would be a
    # second, uncontrolled change stacked on top of the one being ablated.
    if standardise:
        X = np.clip(
            (raw - mu[:, None, :]) / (sd[:, None, :] + STANDARDISE_EPS), -Z_CLIP, Z_CLIP
        )
    else:
        X = raw
    return EmissionSet(
        merchant_ids=merchants,
        window_index=windows,
        X=X,
        feature_names=tuple(feature_names),
        segments=segments,
        burn_in_windows=burn_in_windows,
        shrinkage_weight=weight[:, 0],
        segment_map=segment_map,
        n_txn=n_txn,
    )
