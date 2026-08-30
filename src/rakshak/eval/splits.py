"""The leakage guard — temporal AND merchant-group splits, enforced in code.

`06-requirements.md` §3 freezes the evaluation:

    Split: Temporal — train months 1-6, validate month 7, test months 8-9.
           AND merchant-group — no merchant ID crosses splits.
           Both enforced in `eval/splits.py`.

This module is the single place that decides which rows any model is allowed to
see. If it is wrong, every number in the README is a lie, so the overlap check
is a callable assertion (`assert_no_leakage`) that `load_split` runs on every
call rather than a comment asking future code to behave.

Two design decisions worth stating out loud, because both are questionable and
both are cheap to change later:

1. **A split is the cross-product of a merchant group and a time window.**
   Train = train-group merchants over days [0, 180). Test = test-group
   merchants over days [210, 270). Neither dimension alone is sufficient:
   the merchant group stops fitted per-merchant parameters from crossing over,
   the time window stops the future from leaking into the past.

2. **`Split.transactions` carries each merchant's history from day 0 up to the
   window end, not only the rows inside the window.** A per-merchant sequence
   model has to see a merchant's own past to hold a belief about it. That past
   is not leakage under a merchant-group split — those rows were never in any
   other split, and they are strictly earlier than the decision point. Use
   `Split.window_transactions` when you want the window rows alone.

Merchant-group assignment is deterministic and **does not depend on `--seed`**:
the frozen eval should not move when someone changes a seed. It is stratified
by typology so that each split sees all five typologies (FR-018's per-typology
reporting needs that), by interleaving the sorted merchant IDs within each
typology 3:1:1.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from rakshak.config import (
    BAD_STATES,
    GENERATOR_START_DATE,
    MERCHANT_GROUP_CYCLE,
    SPLIT_DAY_BOUNDS,
    STATE_PATHS_PARQUET,
    TRANSACTIONS_PARQUET,
)
from rakshak.decision.cost import (
    expected_monthly_volume_inr,
    merchant_value_inr,
    realised_loss_inr,
)

SplitName = Literal["train", "validate", "test"]

_TEST_UNLOCK_TICKETS: frozenset[str] = frozenset({"T-0011", "T-0013"})
"""Only the tickets that render the final verdict may open the test window
(06-requirements.md §3: "Test set touched — exactly once, at the end")."""


# ---------------------------------------------------------------------------
# The active dataset (T-0022b)
# ---------------------------------------------------------------------------
#
# T-0022c has to score a second population (`data/synthetic_shock/`) through the
# exact same fitting and scoring code that produced every committed number. But
# `load_split` is not the only reader: `split_summary`, `gbdt.build_window_matrix`
# and `hmm_score.fit` also open the parquets, and the `Scorer` contract
# (`harness.py`) is `(split, rng) -> frame`, so there is no argument channel to
# reach the two `fit()` calls without changing `MODEL_REGISTRY` and every
# `score_*` signature. These two module-level paths are that channel instead.
# See the 2026-08-30 amendment in `11-tickets/T-0022b.md`.
#
# ponytail: module-level mutable state — not thread-safe, and not nestable across
# concurrent runs. Upgrade path if the harness ever scores datasets in parallel:
# thread the paths through `Scorer` and `evaluate_model` instead.

_ACTIVE_TRANSACTIONS: Path = TRANSACTIONS_PARQUET
_ACTIVE_STATE_PATHS: Path = STATE_PATHS_PARQUET


def active_transactions_path() -> Path:
    """Path the current run reads transactions from. Defaults to `data/synthetic/`."""
    return _ACTIVE_TRANSACTIONS


def active_state_paths_path() -> Path:
    """Path the current run reads state paths from. Defaults to `data/synthetic/`."""
    return _ACTIVE_STATE_PATHS


@contextmanager
def active_dataset(
    transactions_path: Path | None = None,
    state_paths_path: Path | None = None,
) -> Iterator[None]:
    """Point every dataset reader at an alternate parquet pair for the duration.

    A dataset override is NOT a test-window unlock: `load_split("test")` still
    raises without `unlock_test=`, whichever dataset is active.

    Args:
        transactions_path: Alternate transactions parquet. None keeps the current one.
        state_paths_path: Alternate state-paths parquet. None keeps the current one.

    Yields:
        None. Restores the previous paths on exit, including on exception.
    """
    global _ACTIVE_TRANSACTIONS, _ACTIVE_STATE_PATHS
    previous = (_ACTIVE_TRANSACTIONS, _ACTIVE_STATE_PATHS)
    if transactions_path is not None:
        _ACTIVE_TRANSACTIONS = Path(transactions_path)
    if state_paths_path is not None:
        _ACTIVE_STATE_PATHS = Path(state_paths_path)
    try:
        yield
    finally:
        _ACTIVE_TRANSACTIONS, _ACTIVE_STATE_PATHS = previous


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def assert_no_leakage(groups: pd.Series) -> None:
    """Fail loudly if any merchant ID is assigned to more than one split.

    NFR-002, zero tolerance. Call this from anywhere that builds or consumes a
    split; `load_split` calls it on every invocation.

    Args:
        groups: Series indexed by merchant_id, values in {"train","validate","test"}.

    Raises:
        AssertionError: if the index has duplicates, if any merchant maps to
            more than one split, or if a split name is unrecognised.
    """
    duplicated = groups.index[groups.index.duplicated()].unique().tolist()
    if duplicated:
        raise AssertionError(
            f"NFR-002 violated: {len(duplicated)} merchant ID(s) assigned more than once, "
            f"e.g. {sorted(duplicated)[:5]}"
        )
    unknown = sorted(set(groups.unique()) - set(SPLIT_DAY_BOUNDS))
    if unknown:
        raise AssertionError(f"unknown split name(s): {unknown}")
    per_merchant = groups.groupby(level=0).nunique()
    crossing = per_merchant[per_merchant > 1].index.tolist()
    if crossing:
        raise AssertionError(
            f"NFR-002 violated: {len(crossing)} merchant ID(s) cross splits, "
            f"e.g. {sorted(crossing)[:5]}"
        )


def assert_window_is_frozen() -> None:
    """Fail if the temporal windows drift from the frozen spec (06-req §3)."""
    expected = {"train": (0, 180), "validate": (180, 210), "test": (210, 270)}
    if SPLIT_DAY_BOUNDS != expected:
        raise AssertionError(f"temporal split drifted from frozen spec: {SPLIT_DAY_BOUNDS}")


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------


def assign_merchant_groups(state_paths: pd.DataFrame) -> pd.Series:
    """Assign every merchant to exactly one split, stratified by typology.

    Deterministic and seed-independent: merchant IDs are sorted within each
    typology and dealt round-robin over `MERCHANT_GROUP_CYCLE`.

    Args:
        state_paths: The generator's state-path frame; needs `merchant_id` and
            `typology` columns.

    Returns:
        Series indexed by merchant_id (sorted), values in {"train","validate","test"}.
    """
    per_merchant = (
        state_paths[["merchant_id", "typology"]]
        .drop_duplicates()
        .sort_values(["typology", "merchant_id"], kind="stable")
    )
    n = len(MERCHANT_GROUP_CYCLE)
    assignments: dict[str, str] = {}
    for _typology, block in per_merchant.groupby("typology", sort=True):
        for position, merchant_id in enumerate(block["merchant_id"]):
            assignments[merchant_id] = MERCHANT_GROUP_CYCLE[position % n]
    groups = pd.Series(assignments, name="split").sort_index()
    groups.index.name = "merchant_id"
    assert_no_leakage(groups)
    return groups


# ---------------------------------------------------------------------------
# The split object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """One evaluation split: a merchant group crossed with a time window.

    Attributes:
        name: "train" | "validate" | "test".
        start_day: Window start, days since GENERATOR_START_DATE (inclusive).
        end_day: Window end, days since GENERATOR_START_DATE (exclusive).
        merchant_ids: Sorted merchant IDs in this split. Disjoint across splits.
        transactions: This split's merchants, all rows with day < end_day —
            i.e. history up to the window end (see module docstring). Carries an
            extra integer `day` column.
        labels: 1 if the merchant enters a state in `BAD_STATES` before
            `end_day`, else 0. Indexed by merchant_id. Units: dimensionless.
        transition_day: Day of the merchant's first bad state, NaN if never bad.
            Ground truth for detection lag. Units: days.
        transition_timestamp: Same, as a timestamp. NaT if never bad.
        loss_inr: L_m — **realised** fraud loss, r_cb * (1 + phi) * gross volume
            transacted while in a bad state before `end_day` (07-math.md §5,
            `decision.cost.realised_loss_inr`). Units: INR. 0 for healthy
            merchants. Turnover while bad is NOT loss — see T-0007a.
        value_inr: V_m — expected remaining **lifetime** gross margin,
            g * v_m * l_m, where v_m is non-refund volume before `end_day`
            annualised to a monthly rate (07-math.md §5,
            `decision.cost.merchant_value_inr`). Units: INR.
    """

    name: str
    start_day: int
    end_day: int
    merchant_ids: tuple[str, ...]
    transactions: pd.DataFrame
    labels: pd.Series
    transition_day: pd.Series
    transition_timestamp: pd.Series
    loss_inr: pd.Series
    value_inr: pd.Series

    @property
    def window_transactions(self) -> pd.DataFrame:
        """Rows inside [start_day, end_day) only, excluding the warm-up history."""
        day = self.transactions["day"]
        return self.transactions[(day >= self.start_day) & (day < self.end_day)]

    @property
    def n_merchants(self) -> int:
        return len(self.merchant_ids)

    @property
    def prevalence(self) -> float:
        """Fraction of merchants labelled bad. Far above real-world base rates
        (config.FRAUD_MERCHANT_RATE == 0.20) — must be printed next to every
        precision-like number this split produces."""
        return float(self.labels.mean()) if self.n_merchants else 0.0


def _add_day_column(transactions: pd.DataFrame) -> pd.DataFrame:
    """Attach an integer `day` column (days since GENERATOR_START_DATE)."""
    origin = pd.Timestamp(GENERATOR_START_DATE)
    out = transactions.copy()
    out["day"] = (out["timestamp"] - origin).dt.days.astype("int64")
    return out


def load_split(
    name: SplitName,
    *,
    unlock_test: str | None = None,
    transactions: pd.DataFrame | None = None,
    state_paths: pd.DataFrame | None = None,
    transactions_path: Path | None = None,
    state_paths_path: Path | None = None,
) -> Split:
    """Load one evaluation split, running the leakage guard on the way.

    Args:
        name: "train", "validate", or "test".
        unlock_test: Required only for `name="test"`. Must be the ticket ID that
            06-requirements.md §3 authorises to touch the test window — one of
            "T-0011" or "T-0013". Everything before those tickets tunes on
            "validate". This is the structural version of "test set touched
            exactly once, at the end".
        transactions: Override for the transactions frame (tests use this).
        state_paths: Override for the state-path frame (tests use this).
        transactions_path: Override for the transactions parquet path (T-0022b).
            None reads `active_transactions_path()`. Ignored if `transactions`
            is given. NOT a test-window unlock — `unlock_test` still applies.
        state_paths_path: Override for the state-paths parquet path (T-0022b).
            None reads `active_state_paths_path()`.

    Returns:
        A populated `Split`.

    Raises:
        PermissionError: if `name="test"` without a valid `unlock_test` ticket.
        AssertionError: if the leakage guard or the frozen-window check fails.
    """
    assert_window_is_frozen()
    if name not in SPLIT_DAY_BOUNDS:
        raise ValueError(f"unknown split {name!r}; expected one of {sorted(SPLIT_DAY_BOUNDS)}")
    if name == "test" and unlock_test not in _TEST_UNLOCK_TICKETS:
        raise PermissionError(
            "The test window is locked (06-requirements.md §3: touched exactly once, at the "
            f"end). Pass unlock_test=<ticket> with one of {sorted(_TEST_UNLOCK_TICKETS)}; "
            "tune on split='validate' until then."
        )

    if transactions is None:
        transactions = pd.read_parquet(transactions_path or active_transactions_path())
    if state_paths is None:
        state_paths = pd.read_parquet(state_paths_path or active_state_paths_path())

    groups = assign_merchant_groups(state_paths)
    assert_no_leakage(groups)

    start_day, end_day = SPLIT_DAY_BOUNDS[name]
    merchant_ids = tuple(groups.index[groups == name])
    members = set(merchant_ids)

    txns = _add_day_column(transactions[transactions["merchant_id"].isin(members)])
    txns = txns[txns["day"] < end_day].sort_values(
        ["merchant_id", "timestamp"], kind="stable"
    )

    index = pd.Index(merchant_ids, name="merchant_id")
    bad = state_paths[
        state_paths["merchant_id"].isin(members)
        & state_paths["state"].isin(BAD_STATES)
        & (state_paths["start_day"] < end_day)
    ]
    first_bad = bad.sort_values(["merchant_id", "start_day"], kind="stable").groupby(
        "merchant_id"
    )

    transition_day = first_bad["start_day"].min().reindex(index).astype("float64")
    transition_timestamp = first_bad["start_timestamp"].min().reindex(index)
    labels = transition_day.notna().astype("int64")

    bad_txn_mask = _bad_transaction_mask(txns, state_paths, members)
    loss = (
        txns.loc[bad_txn_mask, ["merchant_id", "amount"]]
        .groupby("merchant_id")["amount"]
        .sum()
        .reindex(index)
        .fillna(0.0)
    )
    volume = (
        txns.loc[~txns["is_refund"], ["merchant_id", "amount"]]
        .groupby("merchant_id")["amount"]
        .sum()
        .reindex(index)
        .fillna(0.0)
    )

    return Split(
        name=name,
        start_day=start_day,
        end_day=end_day,
        merchant_ids=merchant_ids,
        transactions=txns.reset_index(drop=True),
        labels=labels.rename("label"),
        transition_day=transition_day.rename("transition_day"),
        transition_timestamp=transition_timestamp.rename("transition_timestamp"),
        loss_inr=realised_loss_inr(loss).rename("loss_inr"),
        value_inr=merchant_value_inr(
            expected_monthly_volume_inr(volume, observed_days=end_day)
        ).rename("value_inr"),
    )


def _bad_transaction_mask(
    txns: pd.DataFrame, state_paths: pd.DataFrame, members: set[str]
) -> pd.Series:
    """Boolean mask over `txns`: True where the row falls in a bad-state segment.

    Segments are contiguous per merchant, so a merge-on-merchant plus a day
    range test is enough; the frame is small (500 merchants x <=5 segments).
    """
    segments = state_paths[
        state_paths["merchant_id"].isin(members) & state_paths["state"].isin(BAD_STATES)
    ][["merchant_id", "start_day", "end_day"]]
    if segments.empty:
        return pd.Series(False, index=txns.index)
    merged = (
        txns[["merchant_id", "day"]]
        .reset_index()
        .merge(segments, on="merchant_id", how="inner")
    )
    hit = merged[(merged["day"] >= merged["start_day"]) & (merged["day"] < merged["end_day"])]
    mask = pd.Series(False, index=txns.index)
    mask.loc[hit["index"].unique()] = True
    return mask


def split_summary(
    state_paths: pd.DataFrame | None = None,
    state_paths_path: Path | None = None,
) -> pd.DataFrame:
    """Merchant counts per split and per typology. Used by the harness table.

    Args:
        state_paths: Override for the state-path frame (tests use this).
        state_paths_path: Override for the state-paths parquet path (T-0022b).
            None reads `active_state_paths_path()`.

    Returns:
        DataFrame indexed by typology, columns are split names, values are
        merchant counts, plus a "total" row.
    """
    if state_paths is None:
        state_paths = pd.read_parquet(state_paths_path or active_state_paths_path())
    groups = assign_merchant_groups(state_paths)
    per_merchant = state_paths[["merchant_id", "typology"]].drop_duplicates()
    per_merchant = per_merchant.assign(split=per_merchant["merchant_id"].map(groups))
    table = (
        per_merchant.pivot_table(
            index="typology", columns="split", values="merchant_id", aggfunc="count", fill_value=0
        )
        .reindex(columns=list(SPLIT_DAY_BOUNDS), fill_value=0)
        .sort_index()
    )
    table.loc["TOTAL"] = table.sum()
    return table.astype("int64")
