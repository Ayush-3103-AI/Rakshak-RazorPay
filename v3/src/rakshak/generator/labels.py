"""Delayed, noisy and censored label emission. 08-generator-v2-spec.md §6.

v1 trained against labels the instant the fraud occurred. That measured a system that
cannot exist: a real chargeback arrives weeks to months after the transaction it
disputes, and a merchant that is defrauding the platform today is unlabelled today no
matter how obvious it is. ``label_available_at`` is the field that makes v2 honest, and
this module is the only thing that writes it.

Four states, all required, all produced here:

======================  ================================================  ==============
State                   Condition                                          Row
======================  ================================================  ==============
Observed positive       fraud, disputed, resolves inside the horizon       label=1
Unreported positive     fraud, but ``unreported_rate`` fires               label=0
Censored                ``label_available_at > simulation_end``            label=NULL
False positive label    legit merchant, ``spurious_chargeback_rate``       label=1
======================  ================================================  ==============

The middle two are what make this a weak-supervision problem rather than a clean
classification task. A model that assumes its labels are correct will overfit to the
noise, and the harness is built to be able to show that.

**The hard invariant is ``label_available_at > label_event_at >= drift_onset_at``.** The
first leg is enforced in ``schemas.Label.__post_init__``; the second cannot be, because
``drift_onset_at`` is quarantined and must not be reachable from ``Label``. It is
property-tested across the join in ``tests/unit/test_labels.py``, and it is the reason
the exponential draw below is allowed to return exactly zero: a dispute filed the
instant the drift began is legal, one filed before it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rakshak.generator.config import LabelsConfig
from rakshak.schemas import LabelSource

__all__ = ["NO_TIME", "LabelDraw", "emit_labels"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B1 = npt.NDArray[np.bool_]

#: Sentinel for "no timestamp" inside the numpy arrays. Converted to ``None`` at the
#: ``Label`` boundary — 09-interfaces.md forbids sentinels in the contract, not in the
#: intermediate arrays, and a parallel validity mask would be a second thing to keep in
#: sync with the timestamps.
NO_TIME = np.iinfo(np.int64).min

NS_PER_DAY = 86_400_000_000_000


@dataclass(frozen=True, slots=True)
class LabelDraw:
    """Arrays of length ``n_merchants``, aligned with the merchant index everywhere else.

    ``label`` is float so that NaN can carry "censored"; the engine casts it to a
    nullable ``Int8`` at the polars boundary, which is what ``LABEL_SCHEMA`` declares.
    """

    label: F64
    label_event_ns: I64
    label_available_ns: I64
    source: npt.NDArray[np.str_]
    is_censored: B1
    #: Ground truth, not a label field: fraud that was never disputed. Radioactive —
    #: ``is_unreported`` is in ``RADIOACTIVE_FIELDS`` and lives only in ``ground_truth``.
    is_unreported: B1


def emit_labels(
    rng: np.random.Generator,
    cfg: LabelsConfig,
    *,
    drift_onset_ns: I64,
    sim_start_ns: int,
    sim_end_ns: int,
) -> LabelDraw:
    """Draw a label state per merchant.

    ``drift_onset_ns`` is ``NO_TIME`` for merchants that never turned. Everything is
    vectorised over the population: 10,000 merchants is one draw per distribution, not
    10,000 branches.
    """
    n = drift_onset_ns.size
    is_fraud = drift_onset_ns != NO_TIME

    label = np.zeros(n, dtype=np.float64)
    event_ns = np.full(n, NO_TIME, dtype=np.int64)
    available_ns = np.full(n, NO_TIME, dtype=np.int64)
    source = np.full(n, LabelSource.NONE.value, dtype=object)
    is_censored = np.zeros(n, dtype=bool)

    # Fraud that is never reported. label=0 on a merchant that really did defraud: this
    # is not censoring, it is a *wrong* label, and telling the two apart is the point.
    unreported = is_fraud & (rng.random(n) < cfg.unreported_rate)

    # A chargeback on a merchant that did nothing wrong. label=1 on a good merchant.
    spurious = ~is_fraud & (rng.random(n) < cfg.spurious_chargeback_rate)

    disputed = (is_fraud & ~unreported) | spurious

    # Fraud disputes are dated from drift onset; spurious ones from a uniform point in
    # the simulated window, because there is no onset to date them from.
    origin = np.where(
        is_fraud,
        drift_onset_ns,
        sim_start_ns + (rng.random(n) * (sim_end_ns - sim_start_ns)).astype(np.int64),
    )
    # Exponential may return exactly 0.0, and that is legal: `>= drift_onset_at`.
    to_dispute = rng.exponential(cfg.fraud_to_dispute_mean_days, size=n) * NS_PER_DAY
    lo, hi = cfg.dispute_delay_days
    to_available = rng.uniform(lo, hi, size=n) * NS_PER_DAY

    ev = origin + to_dispute.astype(np.int64)
    av = ev + np.maximum(to_available.astype(np.int64), 1)  # strictly after, always

    event_ns = np.where(disputed, ev, NO_TIME)
    available_ns = np.where(disputed, av, NO_TIME)

    censored = disputed & (av > sim_end_ns)
    is_censored = censored

    label = np.where(disputed & ~censored, 1.0, 0.0)
    label = np.where(censored, np.nan, label)
    source = np.where(disputed, LabelSource.CHARGEBACK.value, LabelSource.NONE.value)

    return LabelDraw(
        label=label,
        label_event_ns=event_ns,
        label_available_ns=available_ns,
        source=np.asarray(source, dtype=np.str_),
        is_censored=is_censored,
        is_unreported=unreported,
    )
