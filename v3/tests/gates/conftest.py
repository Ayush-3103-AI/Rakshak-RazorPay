"""Session fixtures for the parity gates, and the GREEN/RED summary printer.

``make gates`` must print a verdict per gate whether or not the gate passed — a suite
that only speaks when it fails cannot tell you that G1's realised Fano came in at 12.26
against a target of 12.25, which is the number the whole overdispersion argument rests
on. pytest captures stdout, so verdicts are recorded through ``gates_report.record()``
and printed from ``pytest_terminal_summary`` here. That also means a RED gate still
reports its statistic, which the ticket requires of G1, G2 and G5.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl
import pytest
from gates_report import GATE_SEED, RESULTS, START, scenario

from rakshak.generator.engine import GeneratedData, generate


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not RESULTS:
        return
    write = terminalreporter.write_line
    write("")
    write("=" * 78)
    write("GENERATOR PARITY GATES - 08-generator-v2-spec.md section 7")
    write("=" * 78)
    for result in sorted(RESULTS, key=lambda r: r.gate):
        write(f"{result.gate:<26} {result.verdict:<6} {result.statistic}")
        if result.detail:
            write(f"{'':<26} {'':<6} {result.detail}")
    write("=" * 78)
    if any(r.verdict == "RED" and r.gate.startswith(("G3", "G4")) for r in RESULTS):
        write("G3 and G4 are BLOCKING. Nothing proceeds while either is RED.")
    write("")


@pytest.fixture(scope="session")
def gate_data() -> GeneratedData:
    """The reference run: the manifest's own prevalence, confounders on."""
    return generate(scenario(), np.random.default_rng(GATE_SEED))


@pytest.fixture(scope="session")
def null_data() -> GeneratedData:
    """``prevalence = 0``, confounders on. Gate G5's run: every alert is by construction
    a false positive, because there is no fraud in the population to find."""
    return generate(scenario(prevalence=0.0), np.random.default_rng(GATE_SEED + 1))


@pytest.fixture(scope="session")
def merchant_days(gate_data: GeneratedData) -> Iterator[pl.DataFrame]:
    """One row per merchant-day, with the observables a rule detector reads."""
    yield (
        gate_data.transactions.filter(~pl.col("is_refund"))
        .with_columns(day=(pl.col("event_time") - START).dt.total_days().cast(pl.Int64))
        .group_by(["merchant_id", "day"])
        .agg(
            pl.len().alias("txn_count"),
            pl.col("amount_inr").sum().alias("gmv"),
            (pl.col("status") == "failed").mean().alias("fail_rate"),
            pl.col("payer_id").n_unique().alias("payers"),
        )
        .sort(["merchant_id", "day"])
    )
