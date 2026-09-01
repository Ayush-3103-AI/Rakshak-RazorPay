"""``materialise(workers=n)`` must be ``materialise(workers=1)``, to the content hash.

Spreading the replay over processes is a scheduling decision. It is legal only because
per-merchant state never interacts across merchants — ``registry.REGISTRY`` holds
stateless singletons and the mutable part of a replay is ``MerchantState``, which one
merchant owns — and the one cross-merchant step, the cohort residual, is not in the
parallel part at all: it runs afterwards, on the finished cube, in the parent.

The failure this file exists to catch is not "parallel crashed". It is "parallel produced
the right rows in the wrong order": chunks reassembled as they finish rather than by
merchant would give a panel with identical row *count*, identical schema and identical
per-row *content*, and a different hash — so every pre-registered number downstream of
``make features`` would be a number about how the process pool was scheduled that day.
Hence the assertions go all the way to ``sha256(schema + hash_rows())``, the idiom
``GeneratedData.sha256()`` uses and gate G3 compares on.

The chunk size is monkeypatched rather than the population scaled, because what is being
guarded is a *boundary* property (a chunk starting mid-merchant, a chunk's local row index
leaking into the global one) and boundaries are what a small ``PARALLEL_CHUNK`` buys
cheaply. Serial is left on its own batch size, so this is one-chunk serial against
nine-chunk parallel rather than two runs cut the same way.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from rakshak.eval.splits import DEFAULT_BOUNDARIES, SplitBoundaries
from rakshak.generator import engine
from rakshak.generator.config import load_scenario
from rakshak.models import dataset

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenario_v2.yaml"
SEED = 42

#: Enough merchants that the chunking below is genuinely plural, few enough to replay
#: three times in a unit test.
N_MERCHANTS = 60

#: Merchants per parallel task, for this test only. 60 / 7 is nine chunks, so chunks
#: outnumber the worker counts tested and completion order is not task order — which is
#: the whole point.
CHUNK = 7

#: A short horizon: the panel's shape is what is under test, not its length, and replaying
#: 31 epochs instead of 300 is the difference between a unit test and a coffee break. The
#: test split still exists in the geometry and is still never read — ``last_day`` defaults
#: to the last VALIDATION day and ``materialise`` refuses anything past it.
BOUNDARIES = SplitBoundaries(
    origin=DEFAULT_BOUNDARIES.origin, train=(0, 20), val=(21, 30), test=(31, 40)
)

Run = tuple[pl.DataFrame, dict[str, Any]]


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real generated dataset, written once. Not a hand-rolled frame: the thing being
    replayed has to be the thing the pipeline replays, refunds and all."""
    base = load_scenario(CONFIG_PATH)
    config = dataclasses.replace(
        base, population=dataclasses.replace(base.population, n_merchants=N_MERCHANTS)
    )
    out = tmp_path_factory.mktemp("panel_root")
    engine.generate(config, np.random.default_rng(SEED)).write(out)
    return out


@pytest.fixture(scope="module")
def serial(root: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    out = tmp_path_factory.mktemp("serial") / "panel.parquet"
    summary = dataset.materialise(root, out, boundaries=BOUNDARIES, workers=1)
    return pl.read_parquet(out), summary


def _sha(frame: pl.DataFrame) -> str:
    """``GeneratedData.sha256()``'s idiom: schema, then row hashes in row order."""
    digest = hashlib.sha256()
    digest.update(str(frame.schema).encode())
    digest.update(frame.hash_rows().to_numpy().tobytes())
    return digest.hexdigest()


def test_the_fixture_is_not_vacuous(serial: Run) -> None:
    """A panel of zero rows, or a population that fits in one chunk, would make every
    assertion below pass for the wrong reason."""
    frame, summary = serial
    assert frame.height > 0
    assert summary["events_replayed"] > 0
    assert summary["merchants"] == N_MERCHANTS
    assert summary["workers"] == 1
    assert -(-N_MERCHANTS // CHUNK) > 4, "chunks must outnumber the worker counts tested"


@pytest.mark.parametrize("workers", [2, 4])
def test_workers_do_not_change_the_panel(
    workers: int,
    root: Path,
    serial: Run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, expected_summary = serial
    monkeypatch.setattr(dataset, "PARALLEL_CHUNK", CHUNK)
    out = tmp_path / "panel.parquet"
    summary = dataset.materialise(root, out, boundaries=BOUNDARIES, workers=workers)
    actual = pl.read_parquet(out)

    assert actual.height == expected.height
    assert actual.schema == expected.schema, "column names or dtypes moved"
    assert (
        actual.get_column("merchant_id").to_list()
        == expected.get_column("merchant_id").to_list()
    ), "rows were reassembled by completion order, not by merchant"
    assert actual.get_column("day").to_list() == expected.get_column("day").to_list()
    assert actual.equals(expected)
    assert _sha(actual) == _sha(expected)

    assert summary["events_replayed"] == expected_summary["events_replayed"]
    assert summary["rows_by_split"] == expected_summary["rows_by_split"]
    assert summary["state_bytes_p99"] == expected_summary["state_bytes_p99"]
    assert summary["workers"] == workers


def test_row_order_is_merchant_major(serial: Run) -> None:
    """The order parallel reassembly has to reproduce, stated independently of it.

    ``materialise`` walks ``sorted(profiles)`` and emits each merchant's kept epochs in day
    order, so the panel is sorted on ``(merchant_id, day)`` — a total order, since the pair
    is unique. If that ever stops holding, the equality above is comparing two panels that
    are both wrong in the same way."""
    frame, _ = serial
    assert frame.equals(frame.sort(["merchant_id", "day"]))
