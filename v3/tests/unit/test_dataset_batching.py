"""``dataset._merchant_blocks_batched`` must be the unbatched walk, exactly.

The batched reader exists only because collecting the whole in-window stream at the
T-0101 geometry (20,000 x 365) does not fit in memory - it is a memory restructuring and
must not be a behaviour change. If the batch size can move the panel, every number
downstream of ``make features`` is a number about the batch size.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from rakshak.models.dataset import _merchant_blocks, _merchant_blocks_batched

AS_OF = datetime(2026, 12, 31, tzinfo=UTC)


def _stream(path: Path) -> Path:
    """A tiny merchant-major event stream, deliberately written out of order."""
    rows = []
    for m in range(7):
        for e in range(5):
            rows.append(
                {
                    "merchant_id": f"M{m:03d}",
                    "event_id": f"E{m:03d}{e:02d}",
                    "event_time": datetime(2026, 1, 1 + e, 12, tzinfo=UTC),
                }
            )
    pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=3).write_parquet(path)
    return path


def _walk(path: Path, batch: int) -> list[tuple[str, list[str]]]:
    merchants = sorted(pl.read_parquet(path).get_column("merchant_id").unique().to_list())
    return [
        (mid, block.get_column("event_id").to_list())
        for mid, block in _merchant_blocks_batched(path, AS_OF, merchants, batch=batch)
    ]


def test_batch_size_cannot_change_the_replay_order(tmp_path: Path) -> None:
    path = _stream(tmp_path / "transactions.parquet")
    reference = _walk(path, batch=1_000_000)
    assert [mid for mid, _ in reference] == [f"M{m:03d}" for m in range(7)]
    for batch in (1, 2, 3, 7):
        assert _walk(path, batch) == reference


def test_batched_walk_matches_the_unbatched_one(tmp_path: Path) -> None:
    """The batched reader against ``_merchant_blocks`` on the whole frame - the thing it
    replaced."""
    path = _stream(tmp_path / "transactions.parquet")
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("event_time") <= AS_OF)
        .sort(["merchant_id", "event_time", "event_id"])
        .collect()
    )
    unbatched = [
        (mid, block.get_column("event_id").to_list()) for mid, block in _merchant_blocks(frame)
    ]
    assert _walk(path, batch=2) == unbatched


def test_the_window_filter_still_applies_per_batch(tmp_path: Path) -> None:
    path = _stream(tmp_path / "transactions.parquet")
    merchants = sorted(pl.read_parquet(path).get_column("merchant_id").unique().to_list())
    cutoff = datetime(2026, 1, 3, 12, tzinfo=UTC)
    seen = [
        e
        for _, block in _merchant_blocks_batched(path, cutoff, merchants, batch=2)
        for e in block.get_column("event_time").to_list()
    ]
    assert seen and max(seen) <= cutoff
    assert len(seen) == 7 * 3
