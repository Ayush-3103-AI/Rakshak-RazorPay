"""Rakshak feature layer — transaction stream to standardised HMM emissions.

Two steps, deliberately separable so `eval/` can fit a segmentation on training
merchants and apply it to held-out ones:

    1. `build_window_features` — raw per-window aggregates (FR-008, FR-009, FR-010)
    2. `standardise_panel`     — within-merchant standardisation (FR-007) with
                                 shrinkage to an MCC x AOV-band segment (FR-011)

`build_emissions` chains them for the common case.
"""

from __future__ import annotations

import pandas as pd

from rakshak.features.standardise import (
    BURN_IN_WINDOWS,
    MIN_SEGMENT_MERCHANTS,
    SHRINKAGE_N0,
    EmissionSet,
    SegmentMap,
    fit_segment_map,
    standardise_panel,
)
from rakshak.features.windows import (
    BASE_FEATURES,
    FEATURE_UNITS,
    VULCAN_FEATURES,
    WINDOW_DAYS,
    build_window_features,
    window_state_labels,
)

__all__ = [
    "BASE_FEATURES",
    "BURN_IN_WINDOWS",
    "FEATURE_UNITS",
    "MIN_SEGMENT_MERCHANTS",
    "SHRINKAGE_N0",
    "VULCAN_FEATURES",
    "WINDOW_DAYS",
    "EmissionSet",
    "SegmentMap",
    "build_emissions",
    "build_window_features",
    "fit_segment_map",
    "standardise_panel",
    "window_state_labels",
]


def build_emissions(
    transactions: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    burn_in_windows: int = BURN_IN_WINDOWS,
    segment_map: SegmentMap | None = None,
    drop_features: tuple[str, ...] = (),
    standardise: bool = True,
) -> EmissionSet:
    """Build standardised HMM emissions from a raw transaction stream.

    Args:
        transactions: Frame in the generator's `TRANSACTION_COLUMNS` schema; an optional
            `risk_score` column activates the Vulcan-proxy emissions (FR-010).
        window_days: Window length in days.
        burn_in_windows: Leading windows used to fit each merchant's own location and
            scale. Must end strictly before any window being evaluated.
        segment_map: Segmentation fitted on the training population; None fits one here.
        drop_features: Feature names omitted from the emission vector. The default `()`
            keeps every feature `build_window_features` produced, so the shipping
            pipeline is untouched. Used only by `eval/ablations.py` (FR-018), which must
            apply the identical value at fit time and at score time - the scorers raise
            on a feature-name mismatch.
        standardise: When False, the raw per-window features pass straight through with
            no within-merchant location/scale and no segment shrinkage - FR-018's
            standardisation-off ablation. The default True is FR-007's shipping path.

    Returns:
        An `EmissionSet` with ``X`` of shape (M, W, D): dimensionless z-scores when
        `standardise` is True, raw feature units (`FEATURE_UNITS`) when it is False.

    Raises:
        ValueError: If `drop_features` names a feature the panel does not carry.
    """
    panel, feature_names = build_window_features(transactions, window_days=window_days)
    if drop_features:
        unknown = tuple(name for name in drop_features if name not in feature_names)
        if unknown:
            raise ValueError(f"drop_features names features that do not exist: {unknown}")
        feature_names = tuple(name for name in feature_names if name not in drop_features)
    return standardise_panel(
        panel,
        feature_names,
        burn_in_windows=burn_in_windows,
        segment_map=segment_map,
        standardise=standardise,
    )
