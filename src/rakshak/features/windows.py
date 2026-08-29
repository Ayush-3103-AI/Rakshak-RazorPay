"""Raw per-window merchant features — the pre-standardisation half of the emission layer.

Everything here is computed from a merchant's own transaction stream inside a fixed
calendar window (default 7 days). Nothing here is comparable across merchants yet: a
grocer's velocity and a jeweller's velocity live three orders of magnitude apart. Making
them comparable is `standardise.py`'s job (FR-007).

Feature groups:
    behavioural / financial (FR-009) — log ticket size, velocity, refunds, chargebacks,
        hour-of-day entropy, payment-method mix entropy, new-payer ratio
    graph-derived scalars (FR-008, ADR-0002) — payer-set entropy, repeat-payer ratio,
        payer-set Jaccard vs. the previous window, Herfindahl on payer volume. These are
        the CPU stand-in for a GNN; there is no graph library and no GPU anywhere here.
    Vulcan proxy (FR-010) — window mean and p95 of a per-transaction risk score when the
        column is present; omitted and logged when absent.

Windows are absolute calendar windows indexed from a fixed epoch, not per-merchant
relative windows, so a temporal split in `eval/splits.py` can cut on window index.

Units are stated per feature in `FEATURE_UNITS`. All entropies are in nats.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rakshak.config import VULCAN_SCORE_COLUMN, WINDOW_DAYS

LOGGER = logging.getLogger(__name__)

_MID = "merchant_id"
_WIN = "window_index"

BASE_FEATURES: tuple[str, ...] = (
    # FR-009 behavioural / financial
    "log_amount_mean",
    "log_amount_var",
    "log_velocity",
    "refund_ratio",
    "chargeback_ratio",
    "chargeback_lag_days",
    "hour_entropy",
    "method_entropy",
    "new_payer_ratio",
    # FR-008 graph-derived scalars (ADR-0002)
    "payer_entropy",
    "repeat_payer_ratio",
    "payer_jaccard_prev",
    "payer_herfindahl",
    # bookkeeping
    "sparse",
)
"""Feature names in emission-vector order, excluding the optional Vulcan pair."""

VULCAN_FEATURES: tuple[str, ...] = ("vulcan_mean", "vulcan_p95")
"""Appended to `BASE_FEATURES` only when the score column is present (FR-010)."""

FEATURE_UNITS: dict[str, str] = {
    "log_amount_mean": "ln(INR)",
    "log_amount_var": "ln(INR)^2",
    "log_velocity": "ln(1 + transactions/day)",
    "refund_ratio": "dimensionless, [0, 1]",
    "chargeback_ratio": "dimensionless, [0, 1]",
    "chargeback_lag_days": "days from a payer's first transaction to its chargeback",
    "hour_entropy": "nats over 24 hour-of-day bins",
    "method_entropy": "nats over payment methods",
    "new_payer_ratio": "dimensionless, [0, 1], transaction-weighted",
    "payer_entropy": "nats over payer transaction-count shares",
    "repeat_payer_ratio": "dimensionless, [0, 1], payer-set-weighted",
    "payer_jaccard_prev": "dimensionless, [0, 1]",
    "payer_herfindahl": "dimensionless, (0, 1], on payer INR volume shares",
    "sparse": "indicator, 1.0 when the window held zero transactions",
    "vulcan_mean": "same units as the supplied risk score",
    "vulcan_p95": "same units as the supplied risk score",
}

# Features that are undefined (not zero) in an empty window and are therefore
# forward-filled from the previous window rather than zero-filled. 08-pseudocode.md §C:
# "do not drop it, or the sequence indices desynchronise from ground truth".
_FFILL_ON_EMPTY: tuple[str, ...] = (
    "log_amount_mean",
    "log_amount_var",
    "refund_ratio",
    "chargeback_ratio",
    "chargeback_lag_days",
    "hour_entropy",
    "method_entropy",
    "new_payer_ratio",
    "payer_entropy",
    "repeat_payer_ratio",
    "payer_jaccard_prev",
    "payer_herfindahl",
    "vulcan_mean",
    "vulcan_p95",
)


def _entropy_by_window(frame: pd.DataFrame, category: str) -> pd.Series:
    """Shannon entropy of a categorical's within-window distribution, in nats.

    Args:
        frame: Transaction-level frame carrying `merchant_id`, `window_index` and
            `category`.
        category: Column whose per-window share distribution is measured.

    Returns:
        Float64 Series of entropies in nats, indexed by (merchant_id, window_index).
    """
    counts = (
        frame.groupby([_MID, _WIN, category], observed=True, sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    total = counts.groupby([_MID, _WIN], observed=True, sort=False)["n"].transform("sum")
    share = counts["n"].to_numpy(dtype=float) / total.to_numpy(dtype=float)
    counts["h"] = -share * np.log(share)
    return counts.groupby([_MID, _WIN], observed=True, sort=False)["h"].sum()


def _payer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Graph-derived scalar features (FR-008, ADR-0002) for every populated window.

    All four are within-merchant by construction: the generator scopes payer ids to a
    single merchant, so no cross-merchant graph exists and none is assumed.

    Args:
        frame: Transaction-level frame with `merchant_id`, `window_index`, `payer_id`,
            `amount` (INR).

    Returns:
        Frame indexed by (merchant_id, window_index) with columns `payer_entropy`,
        `repeat_payer_ratio`, `payer_jaccard_prev`, `payer_herfindahl` and
        `new_payer_ratio`.
    """
    payer_win = (
        frame.groupby([_MID, _WIN, "payer_id"], observed=True, sort=False)
        .agg(n=("amount", "size"), amt=("amount", "sum"))
        .reset_index()
    )

    n = payer_win["n"].to_numpy(dtype=float)
    grp = payer_win.groupby([_MID, _WIN], observed=True, sort=False)
    n_total = grp["n"].transform("sum").to_numpy(dtype=float)
    amt_total = grp["amt"].transform("sum").to_numpy(dtype=float)

    count_share = n / n_total
    payer_win["h"] = -count_share * np.log(count_share)
    amt_share = payer_win["amt"].to_numpy(dtype=float) / np.maximum(amt_total, 1e-12)
    payer_win["hhi"] = amt_share**2

    # A payer is "new" in the first window it ever appears in for that merchant.
    first_window = payer_win.groupby([_MID, "payer_id"], observed=True, sort=False)[
        _WIN
    ].transform("min")
    payer_win["is_new"] = (payer_win[_WIN] == first_window).to_numpy()
    payer_win["new_txns"] = np.where(payer_win["is_new"], n, 0.0)

    # Jaccard vs. the previous window. Sorting by (merchant, payer, window) puts a
    # payer's consecutive appearances adjacent, so an element of P_w ∩ P_{w-1} is a row
    # whose predecessor shares the same (merchant, payer) and sits exactly one window back.
    payer_win = payer_win.sort_values([_MID, "payer_id", _WIN], kind="stable")
    mid_arr = payer_win[_MID].to_numpy()
    payer_arr = payer_win["payer_id"].to_numpy()
    win_arr = payer_win[_WIN].to_numpy()
    same_payer = (mid_arr == np.roll(mid_arr, 1)) & (payer_arr == np.roll(payer_arr, 1))
    consecutive = same_payer & (win_arr - np.roll(win_arr, 1) == 1)
    consecutive[0] = False
    payer_win["carried"] = consecutive.astype(float)

    agg = payer_win.groupby([_MID, _WIN], observed=True, sort=False).agg(
        payer_entropy=("h", "sum"),
        payer_herfindahl=("hhi", "sum"),
        n_payers=("payer_id", "size"),
        n_new_payers=("is_new", "sum"),
        new_txns=("new_txns", "sum"),
        n_txns=("n", "sum"),
        inter_prev=("carried", "sum"),
    )
    agg["repeat_payer_ratio"] = 1.0 - agg["n_new_payers"] / agg["n_payers"]
    agg["new_payer_ratio"] = agg["new_txns"] / agg["n_txns"]

    # |P_{w-1}| for the union denominator. A window whose predecessor is empty or absent
    # scores Jaccard 0 — the correct "nothing carried over" reading.
    prev_index = pd.MultiIndex.from_arrays(
        [agg.index.get_level_values(_MID), agg.index.get_level_values(_WIN) - 1]
    )
    prev_n = np.nan_to_num(
        agg["n_payers"].reindex(prev_index).to_numpy(dtype=float), nan=0.0
    )
    inter = agg["inter_prev"].to_numpy(dtype=float)
    union = agg["n_payers"].to_numpy(dtype=float) + prev_n - inter
    agg["payer_jaccard_prev"] = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)

    return agg[
        [
            "payer_entropy",
            "repeat_payer_ratio",
            "payer_jaccard_prev",
            "payer_herfindahl",
            "new_payer_ratio",
        ]
    ]


def build_window_features(
    transactions: pd.DataFrame, window_days: int = WINDOW_DAYS
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Aggregate a transaction stream into a dense merchant x window feature panel.

    Every merchant gets every window in ``[0, W)`` whether or not it transacted: an empty
    window is emitted with ``sparse = 1`` and zero velocity, and its undefined
    ratio/entropy features are forward-filled from the previous window rather than
    dropped, so sequence index ``w`` always means calendar window ``w``.

    Args:
        transactions: Frame with at least `merchant_id`, `timestamp` (datetime64),
            `amount` (INR), `payer_id`, `method`, `mcc`, `is_refund`, `is_chargeback`.
            An optional `risk_score` column activates the Vulcan-proxy pair (FR-010).
        window_days: Window length in days. Must be >= 1.

    Returns:
        ``(panel, feature_names)``. ``panel`` is indexed by (merchant_id, window_index),
        sorted ascending, has exactly ``M * W`` rows and no NaNs, and carries one column
        per feature plus `mcc` and `n_txn`. ``feature_names`` gives the emission-vector
        column order.
    """
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1; got {window_days}")

    df = transactions.copy()
    epoch = df["timestamp"].min().normalize()
    day = (df["timestamp"] - epoch).dt.total_seconds().to_numpy() / 86_400.0
    df[_WIN] = (day // window_days).astype(np.int64)
    df["hour"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log(df["amount"].to_numpy(dtype=float))

    has_vulcan = VULCAN_SCORE_COLUMN in df.columns
    if has_vulcan:
        LOGGER.info("FR-010: '%s' present; adding %s", VULCAN_SCORE_COLUMN, VULCAN_FEATURES)
    else:
        LOGGER.info("FR-010: no '%s' column; Vulcan-proxy emissions omitted", VULCAN_SCORE_COLUMN)
    feature_names = BASE_FEATURES + (VULCAN_FEATURES if has_vulcan else ())

    # Chargeback-lag proxy. The generator does not link a chargeback to its original
    # transaction (a documented simplification), so "lag" is how long the charging-back
    # payer had been transacting with the merchant beforehand. Short lags are the
    # bust-out signature; long lags read as ordinary disputes.
    payer_first = df.groupby([_MID, "payer_id"], observed=True, sort=False)["timestamp"].transform(
        "min"
    )
    df["cb_lag"] = np.where(
        df["is_chargeback"].to_numpy(),
        (df["timestamp"] - payer_first).dt.total_seconds().to_numpy() / 86_400.0,
        np.nan,
    )

    agg_spec: dict[str, tuple[str, str]] = {
        "log_amount_mean": ("log_amount", "mean"),
        "log_amount_var": ("log_amount", "var"),
        "refund_ratio": ("is_refund", "mean"),
        "chargeback_ratio": ("is_chargeback", "mean"),
        "chargeback_lag_days": ("cb_lag", "mean"),
        "n_txn": ("amount", "size"),
    }
    if has_vulcan:
        agg_spec["vulcan_mean"] = (VULCAN_SCORE_COLUMN, "mean")
    grouped = df.groupby([_MID, _WIN], observed=True, sort=False)
    panel = grouped.agg(**agg_spec)
    if has_vulcan:
        panel["vulcan_p95"] = grouped[VULCAN_SCORE_COLUMN].quantile(0.95)

    panel["hour_entropy"] = _entropy_by_window(df, "hour")
    panel["method_entropy"] = _entropy_by_window(df, "method")
    panel = panel.join(_payer_features(df))

    # A one-transaction window has zero within-window dispersion, not an unknown one.
    populated = panel["n_txn"] > 0
    panel.loc[populated, "log_amount_var"] = panel.loc[populated, "log_amount_var"].fillna(0.0)
    # A populated window with no chargeback has a lag of "no evidence", which is a real
    # observation, not a missing one — `chargeback_ratio` carries the presence signal.
    # Forward-filling here instead (as an earlier draft did) turns the feature into a
    # monotone "this merchant has ever had a chargeback" latch that never resets, and the
    # HMM spends a whole hidden state on it.
    panel.loc[populated, "chargeback_lag_days"] = panel.loc[
        populated, "chargeback_lag_days"
    ].fillna(0.0)

    # Dense (merchant x window) grid: sequence index must line up with ground truth.
    merchants = np.sort(df[_MID].unique())
    windows = np.arange(int(df[_WIN].max()) + 1)
    grid = pd.MultiIndex.from_product([merchants, windows], names=[_MID, _WIN])
    mcc_by_merchant = df.groupby(_MID, observed=True)["mcc"].first()
    panel = panel.reindex(grid).sort_index()

    panel["n_txn"] = panel["n_txn"].fillna(0.0)
    panel["sparse"] = (panel["n_txn"] == 0).astype(float)
    panel["log_velocity"] = np.log1p(panel["n_txn"].to_numpy(dtype=float) / window_days)

    # chargeback_lag_days is also NaN when a populated window simply had no chargeback;
    # carrying the merchant's previous value is the zero-evidence reading, and the
    # leading 0.0 says "no chargeback has ever been seen".
    fill_cols = [c for c in _FFILL_ON_EMPTY if c in panel.columns]
    panel[fill_cols] = panel[fill_cols].groupby(level=_MID, observed=True).ffill().fillna(0.0)

    panel["mcc"] = mcc_by_merchant.reindex(panel.index.get_level_values(_MID)).to_numpy()
    if panel[list(feature_names)].isna().to_numpy().any():
        raise AssertionError("NaN survived the window panel; a fill rule is missing")
    return panel, feature_names


def window_state_labels(
    state_paths: pd.DataFrame,
    merchant_ids: np.ndarray,
    n_windows: int,
    window_days: int = WINDOW_DAYS,
) -> np.ndarray:
    """Project the generator's day-level ground-truth state path onto the window grid.

    Lives beside the feature builder deliberately: ground truth and emissions must share
    one window convention or every recovery metric is quietly measuring an offset.

    Args:
        state_paths: Generator frame with `merchant_id`, `state`, `start_day`, `end_day`
            (end exclusive).
        merchant_ids: Merchants in the row order to emit, shape (M,).
        n_windows: W, the number of windows in the emission panel.
        window_days: Window length in days; must match the feature build.

    Returns:
        Object array of state-name strings, shape (M, W). A window spanning a transition
        takes its modal day-level state, so the label is the state the merchant spent
        most of that window in.
    """
    n_days = n_windows * window_days
    by_merchant = {mid: i for i, mid in enumerate(merchant_ids)}
    per_day = np.empty((len(merchant_ids), n_days), dtype=object)
    per_day[:] = "HEALTHY"
    for mid, state, start, end in zip(
        state_paths["merchant_id"].to_numpy(),
        state_paths["state"].to_numpy(),
        state_paths["start_day"].to_numpy(),
        state_paths["end_day"].to_numpy(),
        strict=True,
    ):
        row = by_merchant.get(mid)
        if row is not None:
            per_day[row, int(start) : min(int(end), n_days)] = state

    blocks = per_day.reshape(len(merchant_ids), n_windows, window_days)
    labels = np.empty((len(merchant_ids), n_windows), dtype=object)
    for i in range(len(merchant_ids)):
        for w in range(n_windows):
            values, counts = np.unique(blocks[i, w], return_counts=True)
            labels[i, w] = values[int(np.argmax(counts))]
    return labels
