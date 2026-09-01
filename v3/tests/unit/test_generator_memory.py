"""The transaction table is built in merchant blocks. This is the guard that the blocks
are a memory decision and not a data decision.

At 20,000 merchants x 365 days the transaction table is ~62M rows and ~19 GB once polars
has laid out its string columns, so ``generate`` never materialises it: it hands back a
factory that replays it in merchant-contiguous pieces, and ``write``/``sha256`` consume
one piece at a time. Everything below asserts the one property that makes that legal --
**the block size is invisible in the output**: same rows, same order, same schema, same
content hash, whether the table comes out in one piece or in sixty.

The block size is monkeypatched rather than the population scaled, because the failure
mode being guarded is a *boundary* bug (a refund whose capture sits in another block, an
``event_id`` renumbered per block, a per-block sort that does not compose into the global
one) and boundaries are what a small ``_BLOCK_ROWS`` manufactures cheaply.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rakshak.generator import engine
from rakshak.generator.config import ScenarioConfig, load_scenario

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
SEED = 42
#: Small enough to run twice in a unit test, large enough that the merchant blocks below
#: are genuinely plural. ``n_days`` is left at the manifest's value because the config
#: requires ``splits.test_end_day == n_days - 1``.
N_MERCHANTS = 60


@pytest.fixture(scope="module")
def config() -> ScenarioConfig:
    base = load_scenario(CONFIG_PATH)
    return dataclasses.replace(
        base, population=dataclasses.replace(base.population, n_merchants=N_MERCHANTS)
    )


def _generate(config: ScenarioConfig) -> engine.GeneratedData:
    return engine.generate(config, np.random.default_rng(SEED))


def test_block_size_changes_nothing_about_the_output(
    config: ScenarioConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One block versus many: identical frame, identical content hash.

    This is the whole contract. If it ever fails, the generator's output has moved and no
    pre-registered number downstream of it means what it said it meant.
    """
    whole = _generate(config)
    assert len(list(whole.blocks())) == 1, (
        "the fixture population no longer fits in one block, so this test is comparing "
        "many blocks against many blocks and proves nothing"
    )
    expected = whole.transactions
    expected_sha = whole.sha256()

    monkeypatch.setattr(engine, "_BLOCK_ROWS", 4_000)
    split = _generate(config)
    blocks = list(split.blocks())
    assert len(blocks) > 5, f"expected the table to be cut up; got {len(blocks)} block(s)"

    assert split.transactions.equals(expected)
    assert split.transactions.schema == expected.schema
    assert split.sha256() == expected_sha
    assert split.row_counts["transactions"] == expected.height


def test_blocks_are_disjoint_and_in_merchant_order(
    config: ScenarioConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocks partition the merchants, ascending, with no merchant split across two.

    That is what makes concatenating per-block sorts equal to one global sort: the sort
    key leads with ``merchant_id``, which is fixed-width zero-padded and therefore orders
    lexicographically exactly as the merchant index orders numerically.
    """
    monkeypatch.setattr(engine, "_BLOCK_ROWS", 4_000)
    blocks = list(_generate(config).blocks())

    seen: set[str] = set()
    high_water = ""
    for block in blocks:
        ids = set(block["merchant_id"].unique().to_list())
        assert not (ids & seen), "a merchant appears in two blocks"
        seen |= ids
        assert min(ids) > high_water, "blocks are not in ascending merchant order"
        high_water = max(ids)
        assert block.equals(block.sort(["merchant_id", "event_time", "event_id"])), (
            "a block is not sorted on the table's sort key"
        )


def test_event_ids_and_refund_parents_survive_the_split(
    config: ScenarioConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``event_id`` numbers the *unsplit* stream and ``refund_of`` points into it.

    Both are the obvious things to get wrong when a whole-population build becomes a
    per-block one: renumber inside the block and every id collides; re-base the refund
    parent onto the block and every ``refund_of`` points at the wrong capture.
    """
    monkeypatch.setattr(engine, "_BLOCK_ROWS", 4_000)
    frame = _generate(config).transactions

    assert frame["event_id"].n_unique() == frame.height, "event_id is not unique"
    assert sorted(frame["event_id"].to_list()) == [
        f"E{i:011d}" for i in range(frame.height)
    ], "event_id no longer numbers the unsplit stream densely from zero"

    refunds = frame.filter(pl.col("is_refund"))
    assert refunds.height > 0, "no refunds in the fixture; this test would be vacuous"
    joined = refunds.select("merchant_id", "refund_of", "event_time").join(
        frame.select(
            pl.col("event_id").alias("refund_of"),
            pl.col("merchant_id").alias("parent_merchant"),
            pl.col("event_time").alias("parent_time"),
        ),
        on="refund_of",
        how="left",
    )
    assert joined.height == refunds.height
    assert joined["parent_merchant"].null_count() == 0, "a refund_of resolves to nothing"
    assert (joined["parent_merchant"] == joined["merchant_id"]).all(), (
        "a refund was attached to another merchant's capture"
    )
    assert (joined["event_time"] > joined["parent_time"]).all(), (
        "a refund lands at or before its own capture"
    )


def test_written_parquet_round_trips(config: ScenarioConfig, tmp_path: Path) -> None:
    """The streamed parquet reads back as the frame that was streamed into it.

    ``write`` uses pyarrow for the transaction table -- polars has no append -- so the
    nanosecond, tz-aware ``event_time`` contract crosses a writer it did not before.
    Silent coercion to microseconds is exactly the kind of thing that would not raise.
    """
    data = _generate(config)
    paths = data.write(tmp_path)
    written = pl.read_parquet(paths["transactions"])

    assert written.schema == data.transactions.schema
    assert written.equals(data.transactions)
    assert data.row_counts == {
        name: pl.read_parquet(path).height for name, path in paths.items()
    }
