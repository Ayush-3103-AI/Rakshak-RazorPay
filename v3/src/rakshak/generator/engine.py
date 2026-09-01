"""The generator orchestrator — the only public entry point of ``rakshak.generator``.

Composition order, and it is load-bearing::

    lambda(m, d) = base(persona) x persona_shape x typology_multiplier
                                 x day_of_week x confounder_intensity

Each factor comes from a module that cannot see the others. ``personas.py`` does not
know what a typology is; ``typologies.py`` does not know what a confounder is;
``confounders.py`` does not know which merchants are fraudulent. The engine is the only
place the three meet, and it meets them by multiplying. That is what makes the
``prevalence=0, confounders=on`` null run (gate G5) a real test rather than a claim
about the author's intentions.

**Everything is array-shaped.** 10,000 merchants x 180 days is 1.8M merchant-days and
roughly 12M transactions; a per-transaction Python loop would take hours and buy nothing.
The only loops in this file are over the eight personas, the sixteen hour-of-day
profiles, and the handful of merchant-days that carry a Hawkes overlay.

**This module is an evaluation artifact, not a fraud toolkit.** It exists to produce a
labelled benchmark on which a *defensive* system can be measured, under a track that is
strictly defence-only. Nothing here describes how to evade a control; the typologies are
coarse behavioural caricatures calibrated to make detection measurable, at a level of
detail that is already public in the fraud-detection literature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.parquet as pq

from rakshak.generator import personas as personas_mod
from rakshak.generator.arrivals import (
    SECONDS_PER_DAY,
    hawkes_overlay,
    nb_daily_counts,
    nb_fano_for_target,
)
from rakshak.generator.config import ScenarioConfig
from rakshak.generator.confounders import ConfounderLayer, build_layer
from rakshak.generator.labels import NO_TIME, emit_labels
from rakshak.generator.typologies import (
    TypologyAssignment,
    assign_typologies,
    intensity_multiplier,
    per_merchant_field,
    ramp_progress,
)
from rakshak.schemas import (
    CARD_INSTRUMENTS,
    GROUND_TRUTH_SCHEMA,
    HASH_LEN,
    LABEL_SCHEMA,
    PAYOUT_SCHEMA,
    PROFILE_SCHEMA,
    SCHEMA_VERSION,
    TRANSACTION_SCHEMA,
    Instrument,
    PersonaId,
    TxnStatus,
    TypologyId,
)

__all__ = ["GeneratedData", "generate"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B1 = npt.NDArray[np.bool_]

NS_PER_DAY = 86_400_000_000_000
NS_PER_HOUR = 3_600_000_000_000

#: Instrument column order. Fixed here so a categorical draw and its decode cannot
#: disagree; ``Instrument`` is a StrEnum and its declaration order is the contract.
INSTRUMENT_ORDER = list(Instrument)

TABLES = ("transactions", "profiles", "payouts", "labels", "ground_truth")

#: Target rows per transaction block. The transaction table is built, hashed and written
#: one merchant-contiguous block at a time and never materialised whole, because at the
#: pre-registered 20,000 x 365 geometry the whole frame is ~19 GB of polars string views
#: on a 16 GB box. Blocks are cut on merchant boundaries and ``merchant_id`` is a
#: zero-padded fixed-width string, so lexicographic merchant order equals numeric merchant
#: order and concatenating the blocks in order reproduces the whole-frame sort exactly --
#: which is what makes this a memory change and not a data change.
_BLOCK_ROWS = 2_000_000


@dataclass(frozen=True)
class GeneratedData:
    """The five tables, as polars frames conforming to the schemas in ``schemas.py``.

    Four of them are small enough to hold. ``transactions`` is not: 20,000 merchants x
    365 days is ~72M rows and ~19 GB once polars has laid out the string columns, so it
    is carried as ``blocks`` -- a factory that replays the table in merchant-contiguous
    pieces on demand. ``write`` and ``sha256`` consume it one block at a time and never
    hold more than one; the ``transactions`` property materialises the whole thing for
    the tests and the gates, which run at a geometry where that is affordable.
    """

    blocks: Callable[[], Iterator[pl.DataFrame]]
    n_transactions: int
    profiles: pl.DataFrame
    payouts: pl.DataFrame
    labels: pl.DataFrame
    ground_truth: pl.DataFrame

    @cached_property
    def transactions(self) -> pl.DataFrame:
        """The whole transaction table. Only touch this at a geometry that fits in RAM."""
        frames = list(self.blocks())
        if not frames:
            return pl.DataFrame(schema=TRANSACTION_SCHEMA)
        return pl.concat(frames, rechunk=True)

    @property
    def row_counts(self) -> dict[str, int]:
        """Rows per table, without materialising ``transactions``."""
        return {
            name: self.n_transactions if name == "transactions" else getattr(self, name).height
            for name in TABLES
        }

    def write(self, root: Path | str) -> dict[str, Path]:
        """Write one parquet per table. ``ground_truth`` and ``labels`` land in the same
        directory as the rest; the quarantine is enforced by the AST scan on
        ``features/`` and ``models/`` (gate G4), not by hiding the file."""
        out = Path(root)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in TABLES:
            path = out / f"{name}.parquet"
            if name == "transactions":
                self._write_transactions(path)
            else:
                getattr(self, name).write_parquet(path)
            paths[name] = path
        return paths

    def _write_transactions(self, path: Path) -> None:
        """Stream the transaction blocks into one parquet, one row group per block.

        pyarrow rather than ``DataFrame.write_parquet`` only because polars has no
        append: the row values and their order are exactly what a whole-frame write
        would have produced, and the result is one ordinary parquet file either way.
        """
        writer: pq.ParquetWriter | None = None
        try:
            for block in self.blocks():
                table = block.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            pl.DataFrame(schema=TRANSACTION_SCHEMA).write_parquet(path)

    def sha256(self) -> str:
        """A content hash over all five tables. Gate G3 compares two runs on this.

        Hashing the frames rather than the parquet files is deliberate: parquet embeds a
        writer version and can vary its compression blocks, so file bytes would report
        RED for reasons that have nothing to do with an unseeded RNG.

        ``hash_rows`` is row-local, so feeding it the transaction blocks in order gives
        byte-identical input to feeding it the concatenated frame. Asserted in
        ``tests/unit/test_generator_memory.py``.
        """
        digest = hashlib.sha256()
        digest.update(b"transactions")
        schema_written = False
        for block in self.blocks():
            if not schema_written:
                digest.update(str(block.schema).encode())
                schema_written = True
            digest.update(block.hash_rows().to_numpy().tobytes())
        if not schema_written:
            digest.update(str(pl.DataFrame(schema=TRANSACTION_SCHEMA).schema).encode())
        for name in TABLES[1:]:
            frame: pl.DataFrame = getattr(self, name)
            digest.update(name.encode())
            digest.update(str(frame.schema).encode())
            digest.update(frame.hash_rows().to_numpy().tobytes())
        return digest.hexdigest()


def _persona_field(
    config: ScenarioConfig, persona_idx: I64, name: str
) -> npt.NDArray[np.float64]:
    table = np.array(
        [float(getattr(config.personas[p], name)) for p in PersonaId], dtype=np.float64
    )
    return np.asarray(table[persona_idx])


def generate(config: ScenarioConfig, rng: np.random.Generator) -> GeneratedData:
    """Generate the full v2 dataset. Deterministic given ``config`` and ``rng``."""
    n = config.population.n_merchants
    n_days = config.population.n_days
    marks = config.marks
    start = datetime.fromisoformat(config.population.start_date).replace(tzinfo=UTC)
    start_ns = int(start.timestamp()) * 1_000_000_000
    end_ns = start_ns + n_days * NS_PER_DAY

    # ── merchants ────────────────────────────────────────────────────────────
    persona_idx = personas_mod.sample_persona_ids(rng, n, config.personas)
    base_lambda = _persona_field(config, persona_idx, "base_daily_txns")
    amount_mu = _persona_field(config, persona_idx, "amount_mu")
    amount_sigma = _persona_field(config, persona_idx, "amount_sigma")
    cnp_share = _persona_field(config, persona_idx, "cnp_share")
    fail_rate = _persona_field(config, persona_idx, "fail_rate")
    refund_rate = _persona_field(config, persona_idx, "refund_rate")
    refund_latency = _persona_field(config, persona_idx, "refund_latency_hours")
    new_payer_rate = _persona_field(config, persona_idx, "new_payer_rate")
    payer_pool = _persona_field(config, persona_idx, "payer_pool")
    payout_period = _persona_field(config, persona_idx, "payout_period_days")
    payout_drawdown = _persona_field(config, persona_idx, "payout_drawdown")
    is_regular = (
        np.array([config.personas[p].regular_arrivals for p in PersonaId])[persona_idx]
    )

    declarable = sorted(g for g in config.mcc_groups if g != config.mcc_drift_group)
    group_idx = rng.integers(0, len(declarable), size=n)
    mcc_group = np.array(declarable, dtype=object)[group_idx]
    mcc = np.array(
        [
            config.mcc_groups[g][int(u * len(config.mcc_groups[g]))]
            for g, u in zip(mcc_group, rng.random(n), strict=True)
        ],
        dtype=object,
    )
    drift_codes = np.array(config.mcc_groups[config.mcc_drift_group], dtype=object)

    onboarded_ns = start_ns - (
        rng.integers(1, config.population.onboarding_spread_days + 1, size=n).astype(np.int64)
        * NS_PER_DAY
    )
    # Declared GMV is what the merchant *said* at onboarding; actual is what it then did.
    # The gap is v_declared_ratio, the one signal Bumblebee structurally cannot have.
    expected_monthly = base_lambda * 30.0 * np.exp(amount_mu + amount_sigma**2 / 2.0)
    declared_monthly_gmv = np.maximum(
        expected_monthly * np.exp(rng.normal(0.0, config.population.declaration_error_sigma, n)),
        1.0,
    )

    # ── who turns, when ──────────────────────────────────────────────────────
    assignment = assign_typologies(rng, n, config.population.prevalence, config.typologies)
    progress = ramp_progress(assignment, n_days)
    typ_mult = intensity_multiplier(assignment, config.typologies, progress)

    def tfield(name: str, default: float) -> F64:
        return per_merchant_field(assignment, config.typologies, name, default=default)

    t_amount_mu = tfield("amount_mu_shift", 0.0)
    t_amount_sigma = tfield("amount_sigma_shift", 0.0)
    t_fail_add = tfield("fail_rate_add", 0.0)
    t_intl_add = tfield("intl_share_add", 0.0)
    t_cnp_add = tfield("cnp_share_add", 0.0)
    t_refund_add = tfield("refund_rate_add", 0.0)
    t_micro = tfield("micro_share", 0.0)
    t_round = tfield("round_amount_share", 0.0)
    t_payer_conc = tfield("payer_concentration", 0.0)
    t_new_payer = tfield("new_payer_rate_add", 0.0)
    t_bin_pool = tfield("bin_pool", float(marks.bin_pool_global))
    t_mcc_drift = tfield("mcc_drift", 0.0)
    t_payout_urgency = tfield("payout_urgency", 1.0)
    t_loss_fraction = tfield("loss_fraction", 0.0)
    t_hawkes = _typology_flag(assignment, config, "hawkes")
    t_hour_flip = _typology_flag(assignment, config, "hour_flip")
    t_ring = _typology_flag(assignment, config, "ring")
    ring_id = rng.integers(0, marks.ring_count, size=n)

    # ── persona daily shape ──────────────────────────────────────────────────
    shape = np.ones((n, n_days), dtype=np.float64)
    for p_index, pid in enumerate(PersonaId):
        rows = np.flatnonzero(persona_idx == p_index)
        if rows.size:
            shape[rows] = personas_mod.daily_shape(rng, config.personas[pid], rows.size, n_days)

    dow_raw = np.array(config.arrivals.dow_factors, dtype=np.float64)
    dow = dow_raw / dow_raw.mean()
    day_of_week = (start.weekday() + np.arange(n_days)) % 7

    confounders = build_layer(config, persona_idx, base_lambda, cnp_share)

    lam = (
        base_lambda[:, None]
        * shape
        * typ_mult
        * dow[day_of_week][None, :]
        * confounders.intensity
    )

    # ── counts, then expansion to transactions ───────────────────────────────
    # The count process supplies only the dispersion the composed intensity does not
    # already carry. Persona shapes, typology ramps and the confounder layer all make a
    # merchant's own rate non-stationary, and that is variance the NB must not add twice
    # — see nb_fano_for_target. Drawing at a flat 12.25 realised 15.11 and G1 was RED.
    counts = nb_daily_counts(rng, lam, nb_fano_for_target(lam, config.arrivals.target_fano))
    flat = counts.ravel()
    total = int(flat.sum())
    if total == 0:
        raise ValueError(
            "the scenario produced zero transactions — check population.n_days and "
            "personas.*.base_daily_txns"
        )

    md = np.repeat(np.arange(n * n_days, dtype=np.int64), flat)
    t_merchant = (md // n_days).astype(np.int64)
    t_day = (md % n_days).astype(np.int64)
    t_progress = progress.ravel()[md]
    del md

    offsets = np.concatenate(([0], np.cumsum(flat)[:-1]))
    rank = np.arange(total, dtype=np.int64) - np.repeat(offsets, flat)
    group_n = np.repeat(flat, flat).astype(np.float64)

    secs = _within_day_seconds(
        rng,
        config,
        persona_idx=persona_idx,
        t_merchant=t_merchant,
        t_progress=t_progress,
        t_hour_flip=t_hour_flip,
        is_regular=is_regular,
        rank=rank,
        group_n=group_n,
    )
    del rank, group_n

    # ── Hawkes self-excitation, for the bursty typologies only ───────────────
    child_parent, child_secs = _hawkes_children(
        rng, config, assignment, t_hawkes, flat, offsets, secs, n_days
    )

    # ── marks ────────────────────────────────────────────────────────────────
    m_of = t_merchant
    mu_t = amount_mu[m_of] + t_amount_mu[m_of] * t_progress
    sigma_t = np.maximum(amount_sigma[m_of] + t_amount_sigma[m_of] * t_progress, 0.05)
    amount = np.exp(mu_t + sigma_t * rng.standard_normal(total))

    micro_hit = rng.random(total) < t_micro[m_of] * t_progress
    amount = np.where(
        micro_hit, rng.uniform(1.0, marks.micro_amount_max, size=total), amount
    )
    round_values = np.array(marks.round_amount_values, dtype=np.float64)
    round_hit = rng.random(total) < t_round[m_of] * t_progress
    amount = np.where(round_hit, rng.choice(round_values, size=total), amount)
    amount = np.maximum(np.round(amount), 1.0)
    del mu_t, sigma_t, micro_hit, round_hit

    instrument = _draw_instruments(
        rng,
        config,
        confounders,
        persona_idx=persona_idx,
        t_merchant=t_merchant,
        t_day=t_day,
        t_progress=t_progress,
        t_intl_add=t_intl_add,
    )

    cnp_p = np.clip(
        cnp_share[m_of]
        + t_cnp_add[m_of] * t_progress
        + confounders.p5_cnp_add[m_of] * confounders.p5_profile[t_day],
        0.0,
        1.0,
    )
    is_cnp = rng.random(total) < cnp_p
    del cnp_p

    fail_p = np.clip(
        fail_rate[m_of]
        + t_fail_add[m_of] * t_progress
        + confounders.fail_rate_add[m_of, t_day],
        0.0,
        0.95,
    )
    u_status = rng.random(total)
    failed = u_status < fail_p
    pending = (~failed) & (u_status < fail_p + marks.pending_rate)
    del u_status

    bin_pool_size = np.where(t_progress > 0.0, t_bin_pool[m_of], float(marks.bin_pool_global))
    bin_id = (rng.random(total) * bin_pool_size).astype(np.int64)
    del bin_pool_size

    payer_id = _draw_payers(
        rng,
        config,
        payer_pool=payer_pool,
        new_payer_rate=new_payer_rate,
        t_payer_conc=t_payer_conc,
        t_new_payer=t_new_payer,
        t_ring=t_ring,
        ring_id=ring_id,
        t_merchant=t_merchant,
        t_progress=t_progress,
    )
    device_id, ip_id = _draw_devices(
        rng,
        config,
        t_payer_conc=t_payer_conc,
        t_ring=t_ring,
        ring_id=ring_id,
        t_merchant=t_merchant,
        t_progress=t_progress,
    )

    txn_mcc = mcc[m_of].copy()
    drift_hit = rng.random(total) < t_mcc_drift[m_of] * t_progress
    if drift_hit.any():
        picks = rng.integers(0, drift_codes.size, size=int(drift_hit.sum()))
        txn_mcc[drift_hit] = drift_codes[picks]
    del drift_hit

    event_ns = start_ns + t_day * NS_PER_DAY + (secs * 1e9).astype(np.int64)

    # Children inherit the parent's payer, device, instrument and MCC — a retry burst is
    # the same actor trying again — but are micro-amount and overwhelmingly declined.
    # That is what makes f_retry_burst_rate and f_auth_fail_rate_z separate R3 from a
    # merchant that simply got busier.
    n_child = child_parent.size
    # Guarded, not computed-then-discarded: these are ``rng`` draws, and whether a
    # zero-length draw advances the bit generator is not a thing this file should have an
    # opinion about. The old code only reached them when there were children.
    if n_child:
        child_ns = (
            start_ns + t_day[child_parent] * NS_PER_DAY + (child_secs * 1e9).astype(np.int64)
        )
        child_amount = np.maximum(
            np.round(rng.uniform(1.0, marks.micro_amount_max, size=n_child)), 1.0
        )
        child_failed = rng.random(n_child) < 0.5 + 0.5 * fail_p[child_parent]

    merged = _MarkArrays(
        merchant=t_merchant,
        day=t_day,
        event_ns=event_ns,
        amount=amount,
        instrument=instrument,
        is_cnp=is_cnp,
        failed=failed,
        pending=pending,
        bin_id=bin_id,
        payer_id=payer_id,
        device_id=device_id,
        ip_id=ip_id,
        mcc=txn_mcc,
        progress=t_progress,
    )
    # Every per-transaction array is now reachable through ``merged``, and only through
    # ``merged`` once these names are gone. That is not tidiness: at 20,000 x 365 the
    # mark stream is ~5.6 GB, and a second live reference to any column is what turns the
    # child append below into a second whole copy of it.
    del t_merchant, t_day, m_of, event_ns, amount, instrument, is_cnp, failed, pending
    del bin_id, payer_id, device_id, ip_id, txn_mcc, t_progress
    del secs, flat, offsets, counts, lam, shape, typ_mult, fail_p

    if n_child:
        _append_children(merged, child_parent, child_ns, child_amount, child_failed)
        del child_ns, child_amount, child_failed
    del child_parent, child_secs

    # ── refunds ──────────────────────────────────────────────────────────────
    refunds = _draw_refunds(
        rng,
        config,
        merged,
        refund_rate=refund_rate,
        refund_latency=refund_latency,
        t_refund_add=t_refund_add,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    def transaction_blocks() -> Iterator[pl.DataFrame]:
        return _transaction_blocks(config, merged, refunds, n)

    # ── settlement and payouts ───────────────────────────────────────────────
    net = np.where(merged.failed | merged.pending, 0.0, merged.amount)
    gmv_md = (
        np.bincount(merged.merchant * n_days + merged.day, weights=net, minlength=n * n_days)
        .reshape(n, n_days)
        .astype(np.float64)
    )
    if refunds.parent.size:
        gmv_md -= np.bincount(
            refunds.merchant * n_days + refunds.day,
            weights=refunds.amount,
            minlength=n * n_days,
        ).reshape(n, n_days)

    payouts = _build_payout_frame(
        rng,
        config,
        gmv_md,
        progress=progress,
        payout_period=payout_period,
        payout_drawdown=payout_drawdown,
        t_payout_urgency=t_payout_urgency,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    # ── labels and quarantined ground truth ──────────────────────────────────
    drift_onset_ns = np.where(
        assignment.is_fraud, start_ns + assignment.onset_day * NS_PER_DAY, NO_TIME
    )
    draw = emit_labels(
        rng,
        config.labels,
        drift_onset_ns=drift_onset_ns,
        sim_start_ns=start_ns,
        sim_end_ns=end_ns,
        label_resolution_ns=start_ns
        + config.labels.label_resolution_horizon_day * NS_PER_DAY,
    )

    post_onset = np.arange(n_days)[None, :] >= assignment.onset_day[:, None]
    post_onset_gmv = np.where(assignment.is_fraud[:, None] & post_onset, gmv_md, 0.0).sum(axis=1)
    true_loss = np.where(
        assignment.is_fraud,
        np.maximum(t_loss_fraction * post_onset_gmv, config.costs.min_true_loss_inr),
        0.0,
    )

    merchant_ids = _merchant_id_series(n)
    profiles = pl.DataFrame(
        {
            "merchant_id": merchant_ids,
            "onboarded_at": _ts(onboarded_ns),
            "mcc": pl.Series(mcc.tolist(), dtype=pl.String),
            "mcc_group": pl.Series(mcc_group.tolist(), dtype=pl.String),
            "declared_monthly_gmv": declared_monthly_gmv,
            "kyc_tier": rng.integers(1, 4, size=n).astype(np.int32),
            "vintage_months": rng.integers(
                0, config.population.max_vintage_months, size=n
            ).astype(np.int32),
            "city_tier": rng.integers(1, 4, size=n).astype(np.int32),
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int32),
        }
    ).select(list(PROFILE_SCHEMA))

    labels = pl.DataFrame(
        {
            "merchant_id": merchant_ids,
            "label": pl.Series(draw.label).cast(pl.Int8, strict=False),
            "label_event_at": _ts_nullable(draw.label_event_ns),
            "label_available_at": _ts_nullable(draw.label_available_ns),
            "label_source": pl.Series(draw.source.tolist(), dtype=pl.String),
            "is_censored": draw.is_censored,
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int32),
        }
    ).select(list(LABEL_SCHEMA))

    typology_values = np.array([t.value for t in TypologyId] + [None], dtype=object)
    gt_typology = typology_values[
        np.where(assignment.is_fraud, assignment.typology_index, len(TypologyId))
    ]
    persona_values = np.array([p.value for p in PersonaId], dtype=object)
    ground_truth = pl.DataFrame(
        {
            "merchant_id": merchant_ids,
            "persona_id": pl.Series(persona_values[persona_idx].tolist(), dtype=pl.String),
            "risk_typology_id": pl.Series(gt_typology.tolist(), dtype=pl.String),
            "drift_onset_at": _ts_nullable(drift_onset_ns),
            "true_loss_amount_inr": true_loss,
            "is_unreported": draw.is_unreported,
            "schema_version": np.full(n, SCHEMA_VERSION, dtype=np.int32),
        }
    ).select(list(GROUND_TRUTH_SCHEMA))

    return GeneratedData(
        blocks=transaction_blocks,
        n_transactions=int(merged.merchant.size + refunds.parent.size),
        profiles=profiles,
        payouts=payouts,
        labels=labels,
        ground_truth=ground_truth,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _MarkArrays:
    """Flat per-transaction arrays before the polars frame is assembled."""

    merchant: I64
    day: I64
    event_ns: I64
    amount: F64
    instrument: I64
    is_cnp: B1
    failed: B1
    pending: B1
    bin_id: I64
    payer_id: I64
    device_id: I64
    ip_id: I64
    mcc: npt.NDArray[np.object_]
    progress: F64


@dataclass(slots=True)
class _Refunds:
    parent: I64
    merchant: I64
    day: I64
    event_ns: I64
    amount: F64


def _typology_flag(assignment: TypologyAssignment, config: ScenarioConfig, name: str) -> B1:
    table = np.array(
        [bool(getattr(config.typologies[t], name)) for t in TypologyId] + [False], dtype=bool
    )
    idx = np.where(assignment.is_fraud, assignment.typology_index, len(TypologyId))
    return np.asarray(table[idx])


def _merchant_id_series(n: int) -> pl.Series:
    return (
        pl.select(
            "M" + pl.int_range(0, n, eager=True).cast(pl.String).str.zfill(6)
        )
        .to_series()
        .alias("merchant_id")
    )


def _ts(values_ns: I64) -> pl.Series:
    """int64 nanoseconds since epoch -> the contract dtype: tz-aware UTC, nanosecond.

    Via a naive Datetime and an explicit ``replace_time_zone``, not a direct cast. The
    T-101 logbook is emphatic that the tz-aware-UTC convention is not self-enforcing; a
    cast that quietly *localises* rather than *labels* would shift every timestamp by the
    machine's offset and nothing downstream would raise.
    """
    return (
        pl.Series(values_ns, dtype=pl.Int64)
        .cast(pl.Datetime("ns"))
        .dt.replace_time_zone("UTC")
    )


def _ts_nullable(values_ns: I64) -> pl.Series:
    """Same, with ``NO_TIME`` mapped to null. 09-interfaces.md: nullable is explicit."""
    return (
        pl.Series(values_ns, dtype=pl.Int64)
        .to_frame("v")
        .select(
            pl.when(pl.col("v") == NO_TIME)
            .then(None)
            .otherwise(pl.col("v"))
            .cast(pl.Datetime("ns"))
            .dt.replace_time_zone("UTC")
        )
        .to_series()
    )


def _within_day_seconds(
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    persona_idx: I64,
    t_merchant: I64,
    t_progress: F64,
    t_hour_flip: B1,
    is_regular: B1,
    rank: I64,
    group_n: F64,
) -> F64:
    """Per-transaction second-of-day, drawn once per hour-of-day *profile*.

    There are sixteen profiles: eight personas, and eight reversed ones for the R6
    account-takeover flip (the operator changed, so the clock they work on changed too).
    Drawing per profile rather than per transaction is the difference between sixteen
    vectorised calls and twelve million Python-level draws.
    """
    total = t_merchant.size
    secs = np.empty(total, dtype=np.float64)

    regular = is_regular[t_merchant]
    if regular.any():
        idx = np.flatnonzero(regular)
        spacing = SECONDS_PER_DAY / np.maximum(group_n[idx], 1.0)
        secs[idx] = spacing * (rank[idx] + 0.5) + rng.normal(
            0.0, config.marks.regular_jitter_seconds, size=idx.size
        )

    weights = np.array(
        [config.personas[p].hour_weights for p in PersonaId], dtype=np.float64
    )
    profile_table = np.vstack([weights, weights[:, ::-1]])
    flipped = t_hour_flip[t_merchant] & (t_progress >= 1.0)
    profile = persona_idx[t_merchant] + len(PersonaId) * flipped.astype(np.int64)

    for p in range(profile_table.shape[0]):
        idx = np.flatnonzero((profile == p) & ~regular)
        if idx.size == 0:
            continue
        w = profile_table[p]
        hours = rng.choice(24, size=idx.size, p=w / w.sum())
        secs[idx] = hours * 3600.0 + rng.uniform(0.0, 3600.0, size=idx.size)

    return np.clip(secs, 0.0, SECONDS_PER_DAY - 1e-6)


def _hawkes_children(
    rng: np.random.Generator,
    config: ScenarioConfig,
    assignment: TypologyAssignment,
    t_hawkes: B1,
    flat: I64,
    offsets: I64,
    secs: F64,
    n_days: int,
) -> tuple[I64, F64]:
    """Self-excited children for the bursty typologies, per affected merchant-day.

    The only Python loop over merchant-days in this file, and it is bounded by
    ``n_fraud x P(R3) x n_days`` — about 3,300 iterations at the configured scenario, not
    1.8M. Vectorising it would mean reimplementing the branching process; the loop is the
    lazy correct answer at this cardinality.
    """
    arr = config.arrivals
    parents_out: list[I64] = []
    secs_out: list[F64] = []
    for m in np.flatnonzero(t_hawkes):
        for day in range(int(assignment.onset_day[m]), n_days):
            cell = m * n_days + day
            k = int(flat[cell])
            if k < 2:
                continue
            lo = int(offsets[cell])
            parents = secs[lo : lo + k]
            merged = hawkes_overlay(
                rng,
                parents,
                excitation=arr.hawkes_excitation,
                decay_minutes=arr.hawkes_decay_minutes,
                window_minutes=arr.hawkes_window_minutes,
                max_generations=arr.hawkes_max_children,
            )
            if merged.size == parents.size:
                continue
            children = merged[~np.isin(merged, parents)]
            if children.size == 0:
                continue
            # Attribute each child to the parent it followed, so it inherits that
            # payer/device: a retry burst is one actor trying again.
            slot = np.clip(np.searchsorted(parents, children) - 1, 0, k - 1)
            parents_out.append(lo + slot)
            secs_out.append(np.clip(children, 0.0, SECONDS_PER_DAY - 1e-6))
    if not parents_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.concatenate(parents_out).astype(np.int64), np.concatenate(secs_out)


def _draw_instruments(
    rng: np.random.Generator,
    config: ScenarioConfig,
    confounders: ConfounderLayer,
    *,
    persona_idx: I64,
    t_merchant: I64,
    t_day: I64,
    t_progress: F64,
    t_intl_add: F64,
) -> I64:
    """Instrument index per transaction: persona mix, then typology and confounder
    *switches*.

    A switch — "with probability q, this transaction used the other rail instead" — is
    both what a mix shift physically is and the only formulation that stays vectorised.
    The alternative, a per-transaction 7-vector of probabilities, is 12M x 7 float64 and
    would not fit in memory for no gain in fidelity.
    """
    total = t_merchant.size
    out = np.empty(total, dtype=np.int64)
    codes = {inst.value: i for i, inst in enumerate(INSTRUMENT_ORDER)}

    for p_index, pid in enumerate(PersonaId):
        idx = np.flatnonzero(persona_idx[t_merchant] == p_index)
        if idx.size == 0:
            continue
        mix = config.personas[pid].instrument_mix
        probs = np.array([mix.get(inst.value, 0.0) for inst in INSTRUMENT_ORDER])
        out[idx] = rng.choice(len(INSTRUMENT_ORDER), size=idx.size, p=probs / probs.sum())

    intl = codes[Instrument.INTL_CARD.value]
    out = np.where(rng.random(total) < t_intl_add[t_merchant] * t_progress, intl, out)

    if confounders.enabled:
        p3 = confounders.p3_prob[t_merchant] * confounders.p3_profile[t_day]
        out = np.where(rng.random(total) < p3, codes[confounders.p3_target], out)
        p4 = confounders.p4_prob[t_merchant] * confounders.p4_profile[t_day]
        out = np.where(rng.random(total) < p4, codes[confounders.p4_target], out)
    return np.asarray(out, dtype=np.int64)


def _draw_payers(
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    payer_pool: F64,
    new_payer_rate: F64,
    t_payer_conc: F64,
    t_new_payer: F64,
    t_ring: B1,
    ring_id: I64,
    t_merchant: I64,
    t_progress: F64,
) -> I64:
    """Payer ids, in three namespaces that must not collide.

    - **Merchant pool** — ``merchant * stride + k``. ``payer_concentration`` shrinks the
      effective pool with progress, which is what makes ``g_payer_hhi`` move for R5/R8.
    - **Ring pool** — shared across the members of an R7 mule ring. Sharing payers is the
      *only* thing that makes R7 detectable, and it is invisible to any single-merchant
      feature, which is why R7's recall is expected to be poor and reported separately.
    - **Fresh payers** — one id each, above ``payer_id_space``.
    """
    marks = config.marks
    total = t_merchant.size
    stride = int(max(config.personas[p].payer_pool for p in PersonaId))

    eff_pool = np.maximum(
        1.0, payer_pool[t_merchant] * (1.0 - t_payer_conc[t_merchant] * t_progress)
    )
    pool_slot = (rng.random(total) * eff_pool).astype(np.int64)
    out = t_merchant * stride + pool_slot

    ring_base = marks.payer_id_space // 2
    in_ring = t_ring[t_merchant] & (t_progress > 0.0)
    if in_ring.any():
        ring_slot = (rng.random(total) * marks.ring_payer_pool).astype(np.int64)
        out = np.where(
            in_ring, ring_base + ring_id[t_merchant] * marks.ring_payer_pool + ring_slot, out
        )

    new_p = np.clip(
        new_payer_rate[t_merchant] + t_new_payer[t_merchant] * t_progress, 0.0, 1.0
    )
    is_new = rng.random(total) < new_p
    n_new = int(is_new.sum())
    if n_new:
        out = out.copy()
        out[is_new] = marks.payer_id_space + np.arange(n_new, dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def _draw_devices(
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    t_payer_conc: F64,
    t_ring: B1,
    ring_id: I64,
    t_merchant: I64,
    t_progress: F64,
) -> tuple[I64, I64]:
    """Device and IP ids. Concentration shrinks the device pool alongside the payer pool
    — one actor behind many "payers" is what ``g_device_reuse_rate`` is looking for."""
    marks = config.marks
    total = t_merchant.size
    shrink = 1.0 - t_payer_conc[t_merchant] * t_progress

    dev_pool = np.maximum(1.0, marks.device_pool_per_merchant * shrink)
    device = t_merchant * marks.device_pool_per_merchant + (
        rng.random(total) * dev_pool
    ).astype(np.int64)

    in_ring = t_ring[t_merchant] & (t_progress > 0.0)
    if in_ring.any():
        base = marks.payer_id_space // 2
        ring_slot = (rng.random(total) * marks.ring_device_pool).astype(np.int64)
        device = np.where(
            in_ring, base + ring_id[t_merchant] * marks.ring_device_pool + ring_slot, device
        )

    ip = t_merchant * marks.ip_pool_per_merchant + (
        rng.random(total) * marks.ip_pool_per_merchant
    ).astype(np.int64)
    return np.asarray(device, dtype=np.int64), np.asarray(ip, dtype=np.int64)


def _draw_refunds(
    rng: np.random.Generator,
    config: ScenarioConfig,
    marks_arr: _MarkArrays,
    *,
    refund_rate: F64,
    refund_latency: F64,
    t_refund_add: F64,
    start_ns: int,
    end_ns: int,
) -> _Refunds:
    """A refund is a *second row* pointing at the capture it reverses.

    ``amount_inr`` stays positive — ``schemas.Transaction`` carries the sign in
    ``is_refund`` — and is drawn as ``Uniform(refund_min_fraction, 1.0)`` of the
    original, so "a refund never exceeds its capture" is true by construction rather than
    by a downstream check. It is property-tested anyway.
    """
    m = marks_arr.merchant
    eligible = ~marks_arr.failed & ~marks_arr.pending
    p = np.clip(refund_rate[m] + t_refund_add[m] * marks_arr.progress, 0.0, 1.0)
    hit = eligible & (rng.random(m.size) < p)
    parent = np.flatnonzero(hit)
    if parent.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return _Refunds(empty_i, empty_i, empty_i, empty_i, np.empty(0, dtype=np.float64))

    latency_ns = (
        rng.exponential(refund_latency[m[parent]], size=parent.size) * NS_PER_HOUR
    ).astype(np.int64)
    event_ns = marks_arr.event_ns[parent] + np.maximum(latency_ns, 1)
    inside = event_ns < end_ns
    parent = parent[inside]
    event_ns = event_ns[inside]
    if parent.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return _Refunds(empty_i, empty_i, empty_i, empty_i, np.empty(0, dtype=np.float64))

    original = marks_arr.amount[parent]
    amount = np.minimum(
        np.maximum(
            np.round(original * rng.uniform(config.marks.refund_min_fraction, 1.0, parent.size)),
            1.0,
        ),
        original,
    )
    day = ((event_ns - start_ns) // NS_PER_DAY).astype(np.int64)
    return _Refunds(
        parent=parent,
        merchant=marks_arr.merchant[parent],
        day=day,
        event_ns=event_ns,
        amount=amount,
    )


def _transaction_blocks(
    config: ScenarioConfig, marks_arr: _MarkArrays, refunds: _Refunds, n_merchants: int
) -> Iterator[pl.DataFrame]:
    """The transaction table, in merchant-contiguous blocks, in merchant order.

    Each block is sorted on the same key as the whole frame was, and the blocks partition
    the merchants in ascending order, so ``pl.concat`` of the blocks is row-for-row the
    frame the single whole-population sort used to produce. ``event_id`` is derived from
    the row's position in the *unsplit* stream, which is why the global row indices are
    threaded through rather than recomputed per block.
    """
    n_base = int(marks_arr.merchant.size)
    per_merchant = max(1.0, (n_base + refunds.parent.size) / max(n_merchants, 1))
    block = max(1, int(_BLOCK_ROWS / per_merchant))
    for m0 in range(0, n_merchants, block):
        m1 = m0 + block
        base_rows = np.flatnonzero((marks_arr.merchant >= m0) & (marks_arr.merchant < m1))
        ref_rows = np.flatnonzero((refunds.merchant >= m0) & (refunds.merchant < m1))
        if base_rows.size == 0 and ref_rows.size == 0:
            continue
        # A refund carries its capture's merchant, so its parent is always inside the
        # same merchant block -- which is what lets the block be built in isolation. The
        # equality below is that assumption, checked rather than asserted in a comment.
        parent_rows = refunds.parent[ref_rows]
        parent_local = np.searchsorted(base_rows, parent_rows)
        if parent_local.size and not np.array_equal(base_rows[parent_local], parent_rows):
            raise AssertionError(
                f"merchant block [{m0}, {m1}) contains a refund whose capture is outside "
                "it; the block decomposition assumes a refund never crosses merchants"
            )
        yield _build_transaction_frame(
            config,
            _take_marks(marks_arr, base_rows),
            _take_refunds(refunds, ref_rows, parent_local),
            base_rows,
            n_base + ref_rows,
            parent_rows,
        )


def _append_children(
    a: _MarkArrays, parent: I64, event_ns: I64, amount: F64, failed: B1
) -> None:
    """Append the Hawkes retry burst to a mark stream, in place, field by field.

    Children inherit the parent's payer, device, instrument and MCC -- a retry burst is
    the same actor trying again -- but are micro-amount and overwhelmingly declined. That
    is what makes ``f_retry_burst_rate`` and ``f_auth_fail_rate_z`` separate R3 from a
    merchant that simply got busier.

    In place, one field at a time, because the obvious version -- a fresh ``_MarkArrays``
    built from fourteen ``np.concatenate`` calls -- holds two complete copies of the mark
    stream at once. Reassigning the attribute drops the old array immediately, so the
    overshoot is one column rather than the whole table.
    """
    n = parent.size
    a.merchant = np.concatenate([a.merchant, a.merchant[parent]])
    a.day = np.concatenate([a.day, a.day[parent]])
    a.event_ns = np.concatenate([a.event_ns, event_ns])
    a.amount = np.concatenate([a.amount, amount])
    a.instrument = np.concatenate([a.instrument, a.instrument[parent]])
    a.is_cnp = np.concatenate([a.is_cnp, a.is_cnp[parent]])
    a.failed = np.concatenate([a.failed, failed])
    a.pending = np.concatenate([a.pending, np.zeros(n, dtype=bool)])
    a.bin_id = np.concatenate([a.bin_id, a.bin_id[parent]])
    a.payer_id = np.concatenate([a.payer_id, a.payer_id[parent]])
    a.device_id = np.concatenate([a.device_id, a.device_id[parent]])
    a.ip_id = np.concatenate([a.ip_id, a.ip_id[parent]])
    a.mcc = np.concatenate([a.mcc, a.mcc[parent]])
    a.progress = np.concatenate([a.progress, a.progress[parent]])


def _take_marks(a: _MarkArrays, idx: I64) -> _MarkArrays:
    """The subset of a mark stream at ``idx``, field by field."""
    return _MarkArrays(
        merchant=a.merchant[idx],
        day=a.day[idx],
        event_ns=a.event_ns[idx],
        amount=a.amount[idx],
        instrument=a.instrument[idx],
        is_cnp=a.is_cnp[idx],
        failed=a.failed[idx],
        pending=a.pending[idx],
        bin_id=a.bin_id[idx],
        payer_id=a.payer_id[idx],
        device_id=a.device_id[idx],
        ip_id=a.ip_id[idx],
        mcc=a.mcc[idx],
        progress=a.progress[idx],
    )


def _take_refunds(r: _Refunds, idx: I64, parent_local: I64) -> _Refunds:
    """The subset of a refund stream at ``idx``, re-based onto a block.

    ``parent`` becomes an index into the *block's* mark arrays, because that is what the
    inherited marks (instrument, BIN, payer, device, MCC) are looked up through. The
    global parent row -- the one ``refund_of`` names an ``event_id`` from -- is threaded
    separately.
    """
    return _Refunds(
        parent=parent_local,
        merchant=r.merchant[idx],
        day=r.day[idx],
        event_ns=r.event_ns[idx],
        amount=r.amount[idx],
    )


def _build_transaction_frame(
    config: ScenarioConfig,
    marks_arr: _MarkArrays,
    refunds: _Refunds,
    base_rows: I64,
    ref_rows: I64,
    refund_parent_rows: I64,
) -> pl.DataFrame:
    marks = config.marks
    n_base = marks_arr.merchant.size
    n_ref = refunds.parent.size
    row = np.concatenate([base_rows, ref_rows])

    merchant = np.concatenate([marks_arr.merchant, refunds.merchant])
    event_ns = np.concatenate([marks_arr.event_ns, refunds.event_ns])
    amount = np.concatenate([marks_arr.amount, refunds.amount])
    instrument = np.concatenate([marks_arr.instrument, marks_arr.instrument[refunds.parent]])
    is_cnp = np.concatenate([marks_arr.is_cnp, marks_arr.is_cnp[refunds.parent]])
    failed = np.concatenate([marks_arr.failed, np.zeros(n_ref, dtype=bool)])
    pending = np.concatenate([marks_arr.pending, np.zeros(n_ref, dtype=bool)])
    bin_id = np.concatenate([marks_arr.bin_id, marks_arr.bin_id[refunds.parent]])
    payer_id = np.concatenate([marks_arr.payer_id, marks_arr.payer_id[refunds.parent]])
    device_id = np.concatenate([marks_arr.device_id, marks_arr.device_id[refunds.parent]])
    ip_id = np.concatenate([marks_arr.ip_id, marks_arr.ip_id[refunds.parent]])
    mcc = np.concatenate([marks_arr.mcc, marks_arr.mcc[refunds.parent]])
    is_refund = np.concatenate(
        [np.zeros(n_base, dtype=bool), np.ones(n_ref, dtype=bool)]
    )

    instrument_names = np.array([i.value for i in INSTRUMENT_ORDER], dtype=object)
    is_card = np.isin(instrument, [INSTRUMENT_ORDER.index(i) for i in CARD_INSTRUMENTS])
    status = np.where(
        failed,
        TxnStatus.FAILED.value,
        np.where(pending, TxnStatus.PENDING.value, TxnStatus.CAPTURED.value),
    )
    # decline_code is non-None iff FAILED (schemas.Transaction). Cycling the alphabet by
    # row index rather than drawing keeps f_decline_entropy well spread without consuming
    # a 12M-element random draw for a field nothing measures the *distribution* of.
    codes = np.array(marks.decline_codes, dtype=object)
    slot = row % codes.size
    decline_code = np.full(n_base + n_ref, None, dtype=object)
    decline_code[failed] = codes[slot[failed]]

    frame = pl.DataFrame(
        {
            "row": row,
            "merchant": merchant,
            "payer": payer_id,
            "event_time": _ts(event_ns),
            "amount_inr": amount,
            "instrument": pl.Series(instrument_names[instrument].tolist(), dtype=pl.String),
            "is_cnp": is_cnp,
            "is_international": instrument == INSTRUMENT_ORDER.index(Instrument.INTL_CARD),
            "bin_id": bin_id,
            "device_id": device_id,
            "ip_id": ip_id,
            "status": pl.Series(status.tolist(), dtype=pl.String),
            "failed": failed,
            "is_card": is_card,
            "mcc": pl.Series(mcc.tolist(), dtype=pl.String),
            "is_refund": is_refund,
            "refund_parent": np.concatenate(
                [np.full(n_base, -1, dtype=np.int64), refund_parent_rows]
            ),
            "decline_code": pl.Series(decline_code.tolist(), dtype=pl.String),
        }
    )

    frame = frame.with_columns(
        event_id=pl.lit("E") + pl.col("row").cast(pl.String).str.zfill(11),
        merchant_id=pl.lit("M") + pl.col("merchant").cast(pl.String).str.zfill(6),
        payer_id=pl.lit("P") + pl.col("payer").cast(pl.String).str.zfill(12),
        event_date=pl.col("event_time").dt.date(),
        bin_hash=pl.when(pl.col("is_card"))
        .then(pl.lit("b") + pl.col("bin_id").cast(pl.String).str.zfill(HASH_LEN - 1))
        .otherwise(None),
        device_hash=pl.lit("d") + pl.col("device_id").cast(pl.String).str.zfill(HASH_LEN - 1),
        ip_hash=pl.lit("i") + pl.col("ip_id").cast(pl.String).str.zfill(HASH_LEN - 1),
        schema_version=pl.lit(SCHEMA_VERSION, dtype=pl.Int32),
    ).with_columns(
        refund_of=pl.when(pl.col("is_refund"))
        .then(pl.lit("E") + pl.col("refund_parent").cast(pl.String).str.zfill(11))
        .otherwise(None),
    )

    return (
        frame.select(list(TRANSACTION_SCHEMA))
        .sort(["merchant_id", "event_time", "event_id"])
    )


def _build_payout_frame(
    rng: np.random.Generator,
    config: ScenarioConfig,
    gmv_md: F64,
    *,
    progress: F64,
    payout_period: F64,
    payout_drawdown: F64,
    t_payout_urgency: F64,
    start_ns: int,
    end_ns: int,
) -> pl.DataFrame:
    """Payout requests against a T+2 settled balance.

    ``s_payout_freq_z`` and ``s_balance_drawdown`` are among the strongest bust-out
    signals in the register, and both are untestable without this table — which is
    exactly why the spec says not to skip it.

    ponytail: ``balance_before_inr`` is the net settled GMV accrued *since the previous
    payout request*, not a true running ledger; the un-drawn remainder of each payout is
    not carried forward. A real ledger needs a sequential scan over 1.8M merchant-days
    and would change ``s_balance_drawdown`` by roughly the 8-25% that ``payout_drawdown``
    leaves behind. Upgrade to a per-merchant scan if that ratio ever becomes a headline
    number rather than a feature.
    """
    n, n_days = gmv_md.shape
    cycle = config.settlement.cycle_days
    settled = np.zeros_like(gmv_md)
    if cycle < n_days:
        settled[:, cycle:] = gmv_md[:, : n_days - cycle]

    urgency = 1.0 + (t_payout_urgency[:, None] - 1.0) * progress
    request_p = np.clip(urgency / payout_period[:, None], 0.0, 1.0)
    flag = rng.random((n, n_days)) < request_p

    cum = np.cumsum(settled, axis=1)
    rows, cols = np.nonzero(flag)
    if rows.size == 0:
        return pl.DataFrame(schema=PAYOUT_SCHEMA)

    cumv = cum[rows, cols]
    same_merchant = np.concatenate(([False], rows[1:] == rows[:-1]))
    prev = np.where(same_merchant, np.concatenate(([0.0], cumv[:-1])), 0.0)
    balance = cumv - prev

    keep = balance > config.settlement.min_payout_inr
    rows, cols, balance = rows[keep], cols[keep], balance[keep]
    if rows.size == 0:
        return pl.DataFrame(schema=PAYOUT_SCHEMA)

    amount = np.maximum(np.round(balance * payout_drawdown[rows]), 1.0)
    requested_ns = (
        start_ns
        + cols * NS_PER_DAY
        + (rng.random(rows.size) * SECONDS_PER_DAY * 1e9).astype(np.int64)
    )
    settled_ns = requested_ns + cycle * NS_PER_DAY
    accelerated = rng.random(rows.size) < np.clip(
        (urgency[rows, cols] - 1.0) * config.settlement.accelerated_prob_per_urgency, 0.0, 0.9
    )

    return pl.DataFrame(
        {
            "payout_id": pl.Series(np.arange(rows.size, dtype=np.int64), dtype=pl.Int64)
            .cast(pl.String)
            .str.zfill(10)
            .str.replace(r"^", "Y"),
            "merchant_id": pl.Series(rows, dtype=pl.Int64)
            .cast(pl.String)
            .str.zfill(6)
            .str.replace(r"^", "M"),
            "requested_at": _ts(requested_ns),
            "settled_at": _ts_nullable(np.where(settled_ns < end_ns, settled_ns, NO_TIME)),
            "amount_inr": amount,
            "balance_before_inr": np.round(balance),
            "is_accelerated": accelerated,
            "schema_version": np.full(rows.size, SCHEMA_VERSION, dtype=np.int32),
        }
    ).select(list(PAYOUT_SCHEMA)).sort(["merchant_id", "requested_at"])
