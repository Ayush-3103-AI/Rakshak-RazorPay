"""Platform confounders P1-P6 — the v2 contribution. 08-generator-v2-spec.md §4.

These are platform-wide events that move **everyone's** features with zero fraud
occurring: a festival, a gateway outage, a pricing change, a new rail, a regulatory
mandate, macro seasonality. Gate G5 runs the generator at ``prevalence = 0`` with this
layer on and asks whether the detector alerts anyway. If it does, the system is a drift
detector wearing a fraud detector's job title.

**The separation is structural, not conventional.** Nothing in ``personas.py`` or
``typologies.py`` imports this module, and nothing here reads a typology. The layer is
composed by the engine as a final multiplication. That is what makes the
``prevalence=0, confounders=on`` null test meaningful: if the confounder effect were
computed anywhere inside the persona code, "the confounder layer is independent" would
be a claim about the author's intent rather than a property of the program.

Magnitudes are in **sigma units of the merchant's own daily feature**, not relative
units. See ``ConfoundersConfig`` for why: at Fano 12.25 a relative multiplier of 1.15
lands at |z| ~ 0.2 and could never satisfy T-114's gate, while the spec's own table
states the effects as "up 2-4σ". Share-valued features (P3, P4, P5) take their sigma
from a **7-day** window, because that is how ``i_mix_jsd``, ``i_bin_hhi`` and
``i_cnp_share`` are defined in the feature register — a one-day share on a six-
transaction merchant is almost pure noise and no shift would ever clear it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rakshak.generator.config import ScenarioConfig
from rakshak.schemas import ConfounderId, PersonaId

__all__ = [
    "ROLLING_WINDOW_DAYS",
    "ConfounderLayer",
    "ConfounderWindow",
    "build_layer",
    "null_layer",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

#: The window the share-valued features are defined over (07-feature-register.md: "7d
#: instrument mix", "Herfindahl over issuer BINs, 7d"). Sigma for a share confounder is
#: computed against this many days of transactions, not one.
ROLLING_WINDOW_DAYS = 7


@dataclass(frozen=True, slots=True)
class ConfounderWindow:
    """One event occurrence, as a half-open day range. Gate G5 shades exactly these."""

    confounder: ConfounderId
    start_day: int
    end_day: int
    #: The observable the gate should z-score inside this window.
    feature: str


@dataclass(frozen=True, slots=True)
class ConfounderLayer:
    """Everything the engine needs to apply P1-P6, and nothing that identifies a merchant
    as fraudulent. Shapes: ``(n_merchants, n_days)`` for the per-day arrays, ``(n_merchants,)``
    for the per-merchant switch probabilities, ``(n_days,)`` for the day profiles."""

    windows: list[ConfounderWindow]
    #: Multiplier on lambda. P1 festival, P2 outage dip, P6 macro sinusoid.
    intensity: F64
    #: Additive shift on the auth-failure rate. P2 outage, P5 transient.
    fail_rate_add: F64
    #: P3: per-merchant probability of switching an instrument to the cheaper rail,
    #: times a permanent step profile over days.
    p3_prob: F64
    p3_profile: F64
    p3_target: str
    #: P4: the same, on an S-curve, toward a newly launched instrument.
    p4_prob: F64
    p4_profile: F64
    p4_target: str
    #: P5: additive shift on the CNP share, permanent from the mandate day.
    p5_cnp_add: F64
    p5_profile: F64

    @property
    def enabled(self) -> bool:
        return bool(self.windows)


def null_layer(n_merchants: int, n_days: int) -> ConfounderLayer:
    """The layer with every effect off. ``confounders.enabled: false`` and the T-112/T-113
    persona and typology tests use it, so those tests measure one thing at a time."""
    zeros_md = np.zeros((n_merchants, n_days), dtype=np.float64)
    return ConfounderLayer(
        windows=[],
        intensity=np.ones((n_merchants, n_days), dtype=np.float64),
        fail_rate_add=zeros_md,
        p3_prob=np.zeros(n_merchants),
        p3_profile=np.zeros(n_days),
        p3_target="",
        p4_prob=np.zeros(n_merchants),
        p4_profile=np.zeros(n_days),
        p4_target="",
        p5_cnp_add=np.zeros(n_merchants),
        p5_profile=np.zeros(n_days),
    )


def _count_sigma_ratio(base_lambda: F64, target_fano: float, window_days: float) -> F64:
    """sd/mean of a merchant's count *averaged over ``window_days``* = sqrt(fano/(lambda*w)).

    The conversion that makes "3 sigma" mean the same thing for a 0.8/day B2B merchant
    and an 11/day marketplace.

    **The window matters and is set per confounder, to the duration of that event.** A
    five-day festival is read on a five-day window; a six-hour outage is read on one day.
    Taking every sigma at one-day resolution would be defensible arithmetic and
    indefensible modelling: at Fano 12.25 and lambda 6, one *daily* sigma is already 1.4x
    the mean, so a "3 sigma" festival would be a 5x spike and a permanently-on P6 would
    swing platform volume by a factor of twenty. Reading each event on its own duration
    puts P1 at a ~2.9x sale spike and P6 at +/- 65%, which are the magnitudes the words
    "festival" and "macro seasonality" actually denote.
    """
    return np.asarray(
        np.sqrt(target_fano / (np.maximum(base_lambda, 1e-9) * max(window_days, 1.0)))
    )


def _share_sigma(share: F64, base_lambda: F64) -> F64:
    """sd of a share measured over ``ROLLING_WINDOW_DAYS`` days of transactions."""
    n = np.maximum(base_lambda * ROLLING_WINDOW_DAYS, 1.0)
    return np.asarray(np.sqrt(np.clip(share * (1.0 - share), 1e-6, None) / n))


def build_layer(
    config: ScenarioConfig,
    persona_idx: I64,
    base_lambda: F64,
    cnp_share: F64,
) -> ConfounderLayer:
    """Compose P1-P6 into one layer.

    Takes no ``rng``: every confounder schedule is deterministic and stated in the config
    (``days: [46, 152]``, ``day: 60``). That is on purpose — gate G5 must know exactly
    which days to shade, and a randomly placed platform event would make "inside the
    window" a thing the gate has to be *told* by the generator rather than by the config
    both of them read. Merchant-level heterogeneity comes from ``persona_sensitivity``
    and from each merchant's own lambda, which is plenty.
    """
    n_merchants = persona_idx.size
    n_days = config.population.n_days
    cf = config.confounders
    if not cf.enabled:
        return null_layer(n_merchants, n_days)

    order = list(PersonaId)
    sensitivity = np.array([cf.persona_sensitivity[p] for p in order])[persona_idx]
    fano = config.arrivals.target_fano
    days = np.arange(n_days, dtype=np.float64)

    intensity = np.ones((n_merchants, n_days), dtype=np.float64)
    fail_add = np.zeros((n_merchants, n_days), dtype=np.float64)
    windows: list[ConfounderWindow] = []

    # ── P1 festival / sale spike: half-sine bump on volume ────────────────────
    p1 = cf.P1_festival
    for start in p1.days[: p1.count]:
        end = min(n_days, start + p1.duration_days)
        if start >= n_days:
            continue
        phase = np.zeros(n_days)
        span = np.arange(start, end)
        phase[span] = np.sin(np.pi * (span - start + 0.5) / p1.duration_days)
        p1_sigma = _count_sigma_ratio(base_lambda, fano, p1.duration_days)
        intensity += (p1.magnitude_sigma * sensitivity * p1_sigma)[:, None] * phase[None, :]
        windows.append(ConfounderWindow(ConfounderId.P1, start, end, "txn_count"))

    # ── P2 gateway outage: failures up, volume down, for duration_hours ───────
    #
    # The intensity dip is diluted by the outage's share of the day, because volume that
    # does not arrive in six hours is genuinely a quarter of the day's volume. The
    # failure elevation is NOT diluted: it is stated directly in sigma units of the daily
    # auth-fail rate, which is the feature the gate reads. Modelling the six hours at
    # hour resolution would change nothing any daily feature can see.
    p2 = cf.P2_outage
    fail_sigma = _share_sigma(np.full(n_merchants, 0.05), base_lambda) * np.sqrt(
        float(ROLLING_WINDOW_DAYS)
    )
    dip = 1.0 - (1.0 - p2.intensity_multiple) * (p2.duration_hours / 24.0)
    for start in p2.days[: p2.count]:
        if start >= n_days:
            continue
        intensity[:, start] *= dip
        fail_add[:, start] += p2.magnitude_sigma * sensitivity * fail_sigma
        windows.append(ConfounderWindow(ConfounderId.P2, start, start + 1, "auth_fail_rate"))

    # ── P3 fee change: permanent step toward the cheaper rail ─────────────────
    p3 = cf.P3_fee_change
    p3_share = np.full(n_merchants, 0.5)  # the shifted rail's share, worst-case variance
    p3_prob = np.clip(
        p3.magnitude_sigma * sensitivity * _share_sigma(p3_share, base_lambda), 0.0, 1.0
    )
    p3_profile = (days >= p3.day).astype(np.float64)
    windows.append(
        ConfounderWindow(
            ConfounderId.P3, p3.day, min(n_days, p3.day + p3.window_days), "instrument_mix"
        )
    )

    # ── P4 new payment method: S-curve share gain over ramp_days ──────────────
    p4 = cf.P4_new_method
    p4_prob = np.clip(
        p4.magnitude_sigma * sensitivity * _share_sigma(p3_share, base_lambda), 0.0, 1.0
    )
    # Logistic centred on the midpoint of the ramp; 6/ramp puts ~5%..95% inside it.
    p4_profile = 1.0 / (1.0 + np.exp(-6.0 * (days - (p4.day + p4.ramp_days / 2.0)) / p4.ramp_days))
    p4_profile[days < p4.day] = 0.0
    windows.append(
        ConfounderWindow(
            ConfounderId.P4, p4.day, min(n_days, p4.day + p4.ramp_days), "new_instrument_share"
        )
    )

    # ── P5 tokenisation mandate: permanent CNP step + a transient failure bump ─
    p5 = cf.P5_regulatory
    p5_cnp_add = p5.magnitude_sigma * sensitivity * _share_sigma(cnp_share, base_lambda)
    p5_profile = (days >= p5.day).astype(np.float64)
    transient = np.zeros(n_days)
    span = np.arange(p5.day, min(n_days, p5.day + p5.transient_days))
    transient[span] = 1.0 - (span - p5.day) / p5.transient_days
    fail_add += (p5.magnitude_sigma * sensitivity * fail_sigma)[:, None] * transient[None, :]
    windows.append(
        ConfounderWindow(
            ConfounderId.P5, p5.day, min(n_days, p5.day + p5.transient_days), "cnp_share"
        )
    )

    # ── P6 macro seasonality: always on ───────────────────────────────────────
    # Multiplicative-exponential, then normalised to mean 1 over the horizon. An additive
    # `1 + A*sin` would go NEGATIVE for a low-lambda merchant (at Fano 12.25 and
    # lambda 6, one sigma is already 1.4x the mean), and clamping the negative half at
    # zero silently *raises* every merchant's average volume — a platform's macro
    # seasonality that inflates annual GMV by 30% is a bug wearing a confounder's name.
    # The exponential form keeps the peak at the requested sigma and cannot go negative.
    p6 = cf.P6_macro
    wave = np.sin(2.0 * np.pi * days / p6.period_days)
    p6_sigma = _count_sigma_ratio(base_lambda, fano, ROLLING_WINDOW_DAYS)
    a = np.log1p(p6.amplitude_sigma * sensitivity * p6_sigma)[:, None]
    p6_factor = np.exp(a * wave[None, :])
    intensity *= p6_factor / p6_factor.mean(axis=1, keepdims=True)
    # The gate's window is the first peak of the sinusoid, where |z| is largest.
    peak = int(round(p6.period_days / 4.0))
    half = max(1, int(round(p6.period_days / p6.peak_window_divisor)))
    windows.append(
        ConfounderWindow(
            ConfounderId.P6, max(0, peak - half), min(n_days, peak + half + 1), "txn_count"
        )
    )

    # A merchant's intensity can be pushed negative by a large negative sigma excursion
    # on a low-lambda merchant. Clamp at zero: no arrivals is a floor, not an error.
    np.clip(intensity, 0.0, None, out=intensity)

    return ConfounderLayer(
        windows=windows,
        intensity=intensity,
        fail_rate_add=fail_add,
        p3_prob=p3_prob,
        p3_profile=p3_profile,
        p3_target=p3.target_instrument,
        p4_prob=p4_prob,
        p4_profile=p4_profile,
        p4_target=p4.target_instrument,
        p5_cnp_add=p5_cnp_add,
        p5_profile=p5_profile,
    )
