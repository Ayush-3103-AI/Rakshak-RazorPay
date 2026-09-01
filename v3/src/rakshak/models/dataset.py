"""Materialise the feature panel the rungs train and score on.

Lane B built the feature layer and Lane C built the harness, but nothing between them
ever wrote a **matrix**: ``FeatureSpec`` gives one value per merchant per epoch and the
rungs need ``(merchant-day x feature)``. This module is that missing step, and it lives
under ``models/`` because it is a consumer of the frozen feature layer, not part of it.

**The online runner is what materialises.** ``batch()`` recomputes from the whole prefix,
so calling it once per epoch is O(epochs x features x prefix) and does not finish on this
dataset. ``update()`` is O(1) per event, the two are asserted equal to 1e-9 in
``tests/parity/``, and using the streaming runner means the matrix a model trains on is
the matrix production would have served. That is the dual-runner design paying for
itself rather than only being tested.

**Merchants are replayed one at a time.** Per-merchant state never interacts across
merchants, so a merchant-major walk keeps one ``MerchantState`` hot instead of ten
thousand, and the cross-merchant step (the cohort residual) is done afterwards on the
finished value cube, vectorised per epoch.

That same independence is the only reason ``workers`` exists: the replay is dispatched
over contiguous merchant chunks and reassembled **by merchant, never by completion**, so
the row order — and therefore the panel's content hash, which gate G3 compares — does not
depend on how the pool happened to schedule. The residual layer stays in the parent,
after the cube is whole, because it is the one step that reads across merchants.

**The test split is not materialised.** ``last_day`` defaults to the last validation day,
so no event after it is ever read and no epoch after it is ever evaluated. That is a
stronger guarantee than remembering not to look: the rows do not exist.

Prime Directive 3: nothing here names a radioactive field. The panel carries features and
identifiers only - labels and truth are joined in by the CLI, on the eval side of the
quarantine.
"""

from __future__ import annotations

import time as _time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from rakshak.eval.splits import DEFAULT_BOUNDARIES, SplitBoundaries, merchant_fold, split_of_day
from rakshak.features import registry, tier1
from rakshak.features.cohort import CohortAssignment, assign_cohorts, residual_matrix
from rakshak.features.state import MerchantState
from rakshak.schemas import Instrument, MerchantProfile, Split, Transaction, TxnStatus

__all__ = [
    "DEFAULT_PANEL",
    "ID_COLUMNS",
    "RESIDUAL_PREFIX",
    "Panel",
    "base_columns",
    "end_of_day",
    "load_panel",
    "load_profiles",
    "materialise",
    "residual_columns",
    "residual_name",
]

#: Where the materialised panel lands. One file; it is regenerable from seed + config.
DEFAULT_PANEL = Path("data/v2/features.parquet")

#: Non-feature columns on the panel.
ID_COLUMNS = ("merchant_id", "day", "split")

#: Prefix for a cohort-residual column. ``cohort.py`` writes the layer as
#: ``r_f(m, t) = z_f(m, t) - loo_median``; the column name says the same thing.
RESIDUAL_PREFIX = "r_"


def residual_name(feature: str) -> str:
    return f"{RESIDUAL_PREFIX}{feature}"


def base_columns() -> tuple[str, ...]:
    """The Rung-2 feature set: every registered feature, in ``registry.ORDER``."""
    return tuple(registry.ORDER)


def residual_columns() -> tuple[str, ...]:
    """The Rung-3 addition, and nothing else: one residual per flagged base feature."""
    return tuple(
        residual_name(n) for n in registry.ORDER if registry.REGISTRY[n].has_cohort_residual
    )


def end_of_day(day: date) -> datetime:
    """The instant an epoch is evaluated: the last nanosecond of the day, UTC.

    The same definition the parity harness uses. Online and offline reading the epoch
    boundary differently is the most common cause of a spurious parity failure, and of a
    real one, so there is one definition of it per process and this is it.
    """
    return datetime.combine(day, time.max, tzinfo=UTC)


def load_profiles(root: Path) -> dict[str, MerchantProfile]:
    """Read ``profiles.parquet`` into the dataclass the feature layer expects."""
    frame = pl.read_parquet(root / "profiles.parquet")
    return {
        row["merchant_id"]: MerchantProfile(
            merchant_id=row["merchant_id"],
            onboarded_at=row["onboarded_at"],
            mcc=row["mcc"],
            mcc_group=row["mcc_group"],
            declared_monthly_gmv=row["declared_monthly_gmv"],
            kyc_tier=row["kyc_tier"],
            vintage_months=row["vintage_months"],
            city_tier=row["city_tier"],
        )
        for row in frame.iter_rows(named=True)
    }


def _to_transaction(row: dict[str, Any]) -> Transaction:
    return Transaction(
        event_id=row["event_id"],
        merchant_id=row["merchant_id"],
        payer_id=row["payer_id"],
        event_time=row["event_time"],
        event_date=row["event_date"],
        amount_inr=row["amount_inr"],
        instrument=Instrument(row["instrument"]),
        is_cnp=row["is_cnp"],
        is_international=row["is_international"],
        bin_hash=row["bin_hash"],
        device_hash=row["device_hash"],
        ip_hash=row["ip_hash"],
        status=TxnStatus(row["status"]),
        decline_code=row["decline_code"],
        mcc=row["mcc"],
        is_refund=row["is_refund"],
        refund_of=row["refund_of"],
    )


def _merchant_blocks(frame: pl.DataFrame) -> Iterator[tuple[str, pl.DataFrame]]:
    """Contiguous per-merchant slices of a frame already sorted merchant-major.

    ``partition_by`` would materialise every group at once; this walks the run-length
    boundaries instead, so peak memory is one merchant's events rather than all of them.
    """
    ids = frame.get_column("merchant_id").to_numpy()
    if ids.size == 0:
        return
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    bounds = np.r_[starts, ids.size]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        yield str(ids[lo]), frame.slice(int(lo), int(hi - lo))


#: Merchants per scan pass. See ``_merchant_blocks_batched``.
MERCHANT_BATCH = 2_000


def _merchant_blocks_batched(
    path: Path, as_of_max: datetime, merchants: list[str], batch: int = MERCHANT_BATCH
) -> Iterator[tuple[str, pl.DataFrame]]:
    """``_merchant_blocks``, but reading the stream a merchant-batch at a time.

    Collecting the whole in-window stream at once was fine at cycle 1's 10,000 x 180
    (12.9M rows). At the T-0101 geometry (20,000 x 365; ~50M rows in the 0-299 window,
    six of them string columns) it commits well over this machine's 16 GB and thrashes
    the pagefile, so the replay never finishes. Per-merchant state never interacts across
    merchants, so the stream can be read in batches instead: peak memory is one batch,
    at the cost of one parquet scan per batch.

    Order is identical to the unbatched walk — ``merchants`` is sorted and each batch is
    sorted merchant-major — so the panel this produces is the same panel.
    """
    for start in range(0, len(merchants), batch):
        chunk = merchants[start : start + batch]
        frame = (
            pl.scan_parquet(path)
            .filter((pl.col("event_time") <= as_of_max) & pl.col("merchant_id").is_in(chunk))
            .sort(["merchant_id", "event_time", "event_id"])
            .collect()
        )
        yield from _merchant_blocks(frame)


#: Merchants per parallel task. Small enough that ``workers`` collected batches and the
#: results in flight stay well inside this machine's RAM (one chunk's events is ~80 MB,
#: its cube slice ~8 MB); large enough that the per-task profile load and parquet scan
#: disappear — the stream is written merchant-major, so row-group pruning makes a
#: chunk's scan ~0.2 s against the full 1.4 GB file.
PARALLEL_CHUNK = 250


def _replay_chunk(
    task: tuple[Path, tuple[str, ...], tuple[datetime, ...]],
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    """Replay one contiguous merchant chunk into its own slice of the cube.

    This is the body of the serial walk with a subset of merchants in it, and it is a pure
    function of its arguments: ``registry.REGISTRY`` holds stateless singletons and the
    mutable part of a replay is ``MerchantState``, which is per merchant and never read by
    another one. That independence is what makes the dispatch below legal — the
    cross-merchant step (the cohort residual) is not in here, it is done afterwards on the
    finished cube.

    Top level, and picklable in and out, because Windows spawns rather than forks.
    """
    root, merchants, as_ofs = task
    profiles = load_profiles(root)
    tier1.load_profiles(profiles)
    specs = [registry.REGISTRY[name] for name in registry.ORDER]
    index = {m: i for i, m in enumerate(merchants)}

    cube = np.zeros((len(merchants), len(as_ofs), len(specs)), dtype=np.float32)
    active = np.zeros((len(merchants), len(as_ofs)), dtype=bool)
    state_bytes: list[int] = []
    seen = 0
    for merchant_id, block in _merchant_blocks_batched(
        root / "transactions.parquet", as_ofs[-1], list(merchants)
    ):
        row = index[merchant_id]
        state = MerchantState(merchant_id=merchant_id, profile=profiles[merchant_id])
        events = block.iter_rows(named=True)
        pending: Transaction | None = None
        for d, as_of in enumerate(as_ofs):
            while True:
                if pending is None:
                    nxt = next(events, None)
                    if nxt is None:
                        break
                    pending = _to_transaction(nxt)
                if pending.event_time > as_of:
                    break
                for spec in specs:
                    spec.update(spec.state_of(state), pending)
                seen += 1
                active[row, d] = True
                pending = None
            if d and active[row, d - 1]:
                active[row, d] = True
            cube[row, d, :] = [spec.value(spec.state_of(state), as_of) for spec in specs]
        state_bytes.append(state.nbytes())
    return cube, active, state_bytes, seen


def _replay_results(
    tasks: list[tuple[Path, tuple[str, ...], tuple[datetime, ...]]], workers: int
) -> Iterator[tuple[np.ndarray, np.ndarray, list[int], int]]:
    """Chunk results **in task order**, whatever order the workers finish in.

    Row order is the determinism contract (G3 hashes the panel), so reassembly is by
    merchant, never by completion. ``workers=1`` runs in-process — no pool, no pickling,
    the serial path exactly as it was. Above that, at most ``2 * workers`` chunks are ever
    in flight, so the results waiting to be consumed cannot grow to a second copy of the
    cube and take the machine down.
    """
    if workers <= 1:
        yield from map(_replay_chunk, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        queued = iter(tasks)
        flight: deque[Future[tuple[np.ndarray, np.ndarray, list[int], int]]] = deque(
            pool.submit(_replay_chunk, t) for t in islice(queued, 2 * workers)
        )
        while flight:
            done = flight.popleft()
            for t in islice(queued, 1):
                flight.append(pool.submit(_replay_chunk, t))
            yield done.result()


@dataclass(frozen=True, slots=True)
class Panel:
    """The materialised feature matrix, plus the identifiers every metric needs.

    ``columns`` is the model's column order and it is carried with the matrix rather than
    recomputed, because a matrix whose column order is inferred at score time is the
    silent failure ``09-interfaces.md`` §9 exists to prevent.
    """

    merchant_id: np.ndarray
    day: np.ndarray
    split: np.ndarray
    x: np.ndarray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.merchant_id)
        if not all(len(a) == n for a in (self.day, self.split)) or self.x.shape[0] != n:
            raise ValueError("Panel columns must be the same length")
        if self.x.shape[1] != len(self.columns):
            raise ValueError(f"matrix has {self.x.shape[1]} columns, {len(self.columns)} names")

    def select(self, split: Split) -> Panel:
        mask = self.split == split
        return Panel(
            merchant_id=self.merchant_id[mask],
            day=self.day[mask],
            split=self.split[mask],
            x=self.x[mask],
            columns=self.columns,
        )

    def rows(self, mask: np.ndarray) -> Panel:
        return Panel(
            merchant_id=self.merchant_id[mask],
            day=self.day[mask],
            split=self.split[mask],
            x=self.x[mask],
            columns=self.columns,
        )

    def with_columns(self, columns: tuple[str, ...]) -> Panel:
        """Restrict to ``columns``, preserving their given order."""
        index = {name: i for i, name in enumerate(self.columns)}
        missing = [c for c in columns if c not in index]
        if missing:
            raise KeyError(f"panel has no column(s) {missing}")
        take = np.array([index[c] for c in columns], dtype=np.int64)
        return Panel(
            merchant_id=self.merchant_id,
            day=self.day,
            split=self.split,
            x=self.x[:, take],
            columns=columns,
        )

    def column(self, name: str) -> np.ndarray:
        values: np.ndarray = self.x[:, self.columns.index(name)]
        return values


def materialise(
    root: Path,
    out: Path = DEFAULT_PANEL,
    *,
    boundaries: SplitBoundaries = DEFAULT_BOUNDARIES,
    fold_fn: Callable[[str], Split] | None = None,
    last_day: int | None = None,
    workers: int = 1,
    echo: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Replay the event stream and write the ``(merchant-day x feature)`` panel.

    ``last_day`` defaults to the final **validation** day. Nothing after it is read, so
    the test split is not merely skipped - it is never touched by this process.

    ``fold_fn`` overrides which fold a merchant belongs to. Defaults to ``None``, which
    uses ``eval.splits.merchant_fold(m, boundaries)`` exactly as before — zero behaviour
    change for every existing caller. T-0101 (GitHub #34) passes a merchant fold ratio
    independent of ``boundaries``' day-span proportions (60/15/25 vs 65.75/16.44/17.81),
    which ``merchant_fold()`` cannot express since it derives its shares from
    ``boundaries`` itself; ``cli.py`` supplies a sibling function instead of this module
    editing ``eval/splits.py`` to add a second ratio to it.

    ``workers`` spreads the replay over processes, one contiguous merchant chunk at a
    time. It defaults to 1 — the serial walk, unchanged, and still the path that has to
    stay reachable — because this panel is what every rung's numbers are about.
    ``tests/unit/test_dataset_parallel.py`` holds any worker count to the serial panel,
    row hash included.
    """
    horizon = boundaries.val[1] if last_day is None else last_day
    if horizon >= boundaries.test[0]:
        raise ValueError(
            f"last_day={horizon} reaches the test split (day {boundaries.test[0]}+). "
            "Materialising test rows is not a thing this module does; the split is opened "
            "once, in T-151, through the CLI's lock guard."
        )

    profiles = load_profiles(root)
    tier1.load_profiles(profiles)
    base = base_columns()
    residual = residual_columns()
    resid_index = np.array([base.index(n[len(RESIDUAL_PREFIX) :]) for n in residual])

    merchants = sorted(profiles)
    days = list(range(horizon + 1))
    as_ofs = [end_of_day(boundaries.origin + timedelta(days=d)) for d in days]

    # (merchants x epochs x features). float32 halves the peak and is what LightGBM bins
    # to anyway; the residual layer is computed in float64 and cast on the way back in.
    cube = np.zeros((len(merchants), len(days), len(base)), dtype=np.float32)
    active = np.zeros((len(merchants), len(days)), dtype=bool)
    state_bytes: list[int] = []

    # Serial keeps the batch size it has always had; a parallel run needs the smaller
    # chunk so that `workers` collected batches fit in RAM at once.
    size = MERCHANT_BATCH if workers <= 1 else PARALLEL_CHUNK
    chunks = [tuple(merchants[i : i + size]) for i in range(0, len(merchants), size)]
    frozen_as_ofs = tuple(as_ofs)
    tasks = [(root, chunk, frozen_as_ofs) for chunk in chunks]

    started = _time.perf_counter()
    seen = 0
    row = 0
    for chunk, (c_cube, c_active, c_bytes, c_seen) in zip(
        chunks, _replay_results(tasks, workers), strict=True
    ):
        cube[row : row + len(chunk)] = c_cube
        active[row : row + len(chunk)] = c_active
        state_bytes.extend(c_bytes)
        seen += c_seen
        row += len(chunk)
        if echo is not None:
            echo(
                f"  {row}/{len(merchants)} merchants, {seen:,} events, "
                f"{_time.perf_counter() - started:.0f}s"
            )

    # The cohort-residual layer, one epoch at a time over the finished cube. Merchants
    # with no events yet are excluded rather than passed in at 0.0: cohort.residuals()
    # drops them for the same reason, and stuffing them in at zero drags the cohort median
    # toward zero on exactly the days a confounder is lifting everyone.
    assignment: CohortAssignment = assign_cohorts(profiles)
    resid = np.zeros((len(merchants), len(days), len(residual)), dtype=np.float32)
    for d in range(len(days)):
        rows = np.flatnonzero(active[:, d])
        if rows.size == 0:
            continue
        present = [merchants[i] for i in rows]
        z = cube[rows, d, :][:, resid_index].astype(np.float64)
        resid[rows, d, :] = residual_matrix(assignment, present, z).astype(np.float32)

    # Keep only rows where the merchant's fold and the day's span agree - the strict
    # reading of "both split constraints simultaneously" that splits.assign_rows applies.
    fold_of = fold_fn if fold_fn is not None else (lambda m: merchant_fold(m, boundaries))
    fold = np.array([fold_of(m) for m in merchants])
    day_split = np.array([split_of_day(d, boundaries) or "" for d in days])
    keep_rows, keep_cols = np.nonzero(fold[:, None] == day_split[None, :])

    panel = pl.DataFrame(
        {
            "merchant_id": [merchants[i] for i in keep_rows],
            "day": keep_cols.astype(np.int16),
            "split": day_split[keep_cols],
        }
    ).with_columns(
        [pl.Series(name, cube[keep_rows, keep_cols, j]) for j, name in enumerate(base)]
        + [pl.Series(name, resid[keep_rows, keep_cols, j]) for j, name in enumerate(residual)]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out)

    return {
        "rows": panel.height,
        "merchants": len(merchants),
        "epochs": len(days),
        "last_day": horizon,
        "events_replayed": seen,
        "workers": workers,
        "seconds": round(_time.perf_counter() - started, 1),
        "base_features": len(base),
        "residual_features": len(residual),
        "state_bytes_p99": float(np.percentile(state_bytes, 99)) if state_bytes else 0.0,
        "rows_by_split": {
            str(k): int(v) for k, v in panel.group_by("split").len().sort("split").iter_rows()
        },
    }


def load_panel(path: Path = DEFAULT_PANEL) -> Panel:
    """Read the materialised panel back as a ``Panel``."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make features` first - the rungs score a materialised "
            "panel, not the raw event stream."
        )
    frame = pl.read_parquet(path)
    columns = tuple(c for c in frame.columns if c not in ID_COLUMNS)
    return Panel(
        merchant_id=frame.get_column("merchant_id").to_numpy(),
        day=frame.get_column("day").to_numpy().astype(np.int64),
        split=frame.get_column("split").to_numpy(),
        x=frame.select(columns).to_numpy().astype(np.float64),
        columns=columns,
    )
