"""Typed configuration for the v2 generator. Every number the generator uses lives here.

CLAUDE.md is explicit: no magic numbers in ``src/``. That rule only holds if the config
surface is *complete* — a dataclass with sensible defaults is a magic number with a
nicer name, because nothing forces the YAML to state it and nothing shows it in a diff.
So every field below is **required**: ``configs/scenario_v2.yaml`` must name it, and a
missing key raises with the dotted path of the field that is missing.

The loader is deliberately strict in both directions. An unknown key is an error too,
because the failure mode that costs a day is a renamed parameter that the YAML still
sets under its old name while the code quietly uses its default.

This module is pure configuration: it holds no RNG, draws nothing, and imports nothing
from the rest of the generator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from rakshak.schemas import ConfounderId, PersonaId, TypologyId

__all__ = [
    "SHARE_TOLERANCE",
    "ArrivalsConfig",
    "CapacityConfig",
    "ConfigError",
    "ConfoundersConfig",
    "CostsConfig",
    "LabelsConfig",
    "MarksConfig",
    "P1Festival",
    "P2Outage",
    "P3FeeChange",
    "P4NewMethod",
    "P5Regulatory",
    "P6Macro",
    "PersonaParams",
    "PopulationConfig",
    "ScenarioConfig",
    "SettlementConfig",
    "TypologyParams",
    "load_scenario",
]

#: Shares are asserted to sum to 1.0 within this. Float64 addition of eight two-decimal
#: literals does not land on exactly 1.0, so an exact test would fail for a reason that
#: has nothing to do with the config being wrong.
SHARE_TOLERANCE = 1e-9


class ConfigError(ValueError):
    """A scenario file is malformed. The message always names the dotted field path.

    Ticket T-110's ``Done when`` clause is specifically that an invalid config raises
    with a message naming the field — a loader that says "KeyError: 'share'" against a
    file with sixty ``share`` keys has told you nothing.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Leaf sections
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    n_merchants: int
    n_days: int
    #: Merchant-level positive rate. BAF-native 1.47%. AP-06: NEVER evaluate at 0.20.
    prevalence: float
    #: Simulation day 0, ISO date. Parsed as UTC midnight — 09-interfaces.md conventions.
    start_date: str
    #: Onboarding is staggered backwards from day 0 over this many days, so that
    #: ``p_days_since_onboarding`` is not the same number for every merchant. Every
    #: merchant is already approved on day 0 — this is a post-onboarding sentinel.
    onboarding_spread_days: int
    #: Lognormal sigma of declared-vs-actual monthly GMV. This is the entire content of
    #: ``v_declared_ratio``: at 0 every merchant declares exactly what it does and the
    #: feature is a constant.
    declaration_error_sigma: float
    #: Business age at onboarding is drawn uniformly on [0, this).
    max_vintage_months: int


@dataclass(frozen=True, slots=True)
class ArrivalsConfig:
    #: Fano = Var/Mean of daily counts. 12.25 measured in v1; Poisson is 1.0.
    target_fano: float
    #: Branching ratio of the Hawkes overlay. Must stay < 1 or the process explodes.
    hawkes_excitation: float
    #: Exponential decay constant of the self-excitation kernel.
    hawkes_decay_minutes: float
    #: Children beyond this horizon are not generated — the kernel has decayed to noise
    #: and the tail costs more than it contributes.
    hawkes_window_minutes: float
    #: Hard cap on generations, so a mis-set excitation fails loudly instead of hanging.
    hawkes_max_children: int
    #: 7 multiplicative day-of-week factors, Monday first. Normalised to mean 1.0.
    dow_factors: list[float]


@dataclass(frozen=True, slots=True)
class PersonaParams:
    """One legitimate behaviour (L1-L8). See 08-generator-v2-spec.md §2.

    The hard-negative annotations in that table are the reason most of these fields
    exist: ``refund_rate`` is here so L8 can break the F6 family, ``regular_arrivals`` so
    L5 can look scripted without being fraudulent, ``new_payer_rate`` so L6 can look like
    a churned payer base.
    """

    share: float
    #: Mean daily transactions before any shape, typology or confounder multiplier.
    base_daily_txns: float
    #: LogNormal(mu, sigma) over whole rupees. Multiplicative and heavy-tailed, which is
    #: what payment amounts are; Gaussian amounts make t_wasserstein_7d trivial.
    amount_mu: float
    amount_sigma: float
    refund_rate: float
    refund_latency_hours: float
    #: Probability a transaction comes from a payer never seen before.
    new_payer_rate: float
    #: Size of the recurring payer pool a merchant draws its non-new payers from.
    payer_pool: int
    #: L5 only: arrivals evenly spaced through the active window instead of drawn from
    #: the hour-of-day categorical. This is what drives h_interarrival_cv below 0.3.
    regular_arrivals: bool
    #: Mean days between payout requests.
    payout_period_days: float
    #: Share of balance withdrawn per payout request.
    payout_drawdown: float
    #: One of: flat, seasonal, growth, lumpy, dormant.
    shape: str
    #: Meaning depends on shape: seasonal peak multiple, growth end multiple, or the
    #: dormancy depth. Ignored by flat.
    shape_strength: float
    #: Fraction of the window the shape's special regime occupies (seasonal peak width,
    #: dormancy length, lumpy burst probability), as a share of n_days.
    shape_span: float
    #: Number of shape occurrences: sale windows for seasonal. Ignored by the others.
    shape_count: int
    #: Lognormal sigma on the per-merchant endpoint of a growth ramp. Varies the *level*
    #: and not the curvature, so the L3 cohort mean stays linear — which is the whole
    #: basis of the L3-vs-R1 separation.
    shape_jitter: float
    #: Baseline mark distributions. International share is carried by the intl_card
    #: entry of instrument_mix, not by a separate field, so the two cannot disagree.
    cnp_share: float
    fail_rate: float
    #: Instrument mix, keyed by ``Instrument`` value. Must sum to 1.0.
    instrument_mix: dict[str, float]
    #: 24 hour-of-day weights, normalised internally. Shape, not level.
    hour_weights: list[float]


@dataclass(frozen=True, slots=True)
class TypologyParams:
    """One named fraud pattern (R1-R9). See 08-generator-v2-spec.md §3.

    A fraud merchant is a legitimate merchant that turned: it keeps its persona for the
    pre-onset stream and layers these deltas on top from ``drift_onset_at`` onward.
    """

    #: Share of the positive class. The nine values must sum to 1.0.
    mix: float
    onset_day_min: int
    onset_day_max: int
    ramp_days_min: int
    ramp_days_max: int
    #: Intensity multiple reached at the end of the ramp.
    intensity_multiple: float
    #: Ramp exponent. 1.0 is linear; > 1 is convex, which is what separates R1 from L3.
    convexity: float
    #: R1 only: intensity collapses after the ramp. The bust half of bust-out.
    vanish_after_ramp: bool
    #: The residual intensity multiple once vanished. Ignored when vanish_after_ramp is
    #: false, but still required in the YAML — an ignored-but-stated field is auditable
    #: and an omitted one is a default.
    vanish_intensity: float
    #: Additive/multiplicative mark deltas, all applied at full ramp and scaled down the
    #: ramp so a slow typology (R2) never jumps.
    amount_mu_shift: float
    amount_sigma_shift: float
    fail_rate_add: float
    intl_share_add: float
    cnp_share_add: float
    refund_rate_add: float
    #: Share of post-onset transactions that are sub-₹10 probes (R3).
    micro_share: float
    #: Share of post-onset transactions snapped to a round value (R3, R4).
    round_amount_share: float
    #: 0 keeps the persona payer pool; 1 collapses it to a handful of repeat payers (R5, R8).
    payer_concentration: float
    new_payer_rate_add: float
    #: Distinct issuer BINs the typology draws from. Small = concentrated stolen source.
    bin_pool: int
    #: Turn on the Hawkes self-excitation overlay (R3 card testing).
    hawkes: bool
    #: R6: hour-of-day histogram is reversed overnight at onset.
    hour_flip: bool
    #: R9: probability a post-onset transaction carries an MCC outside the declared group.
    mcc_drift: float
    #: Multiplier on payout request frequency after onset (F8 family).
    payout_urgency: float
    #: R7: draws payers and devices from a shared ring pool rather than its own.
    ring: bool
    #: true_loss_amount_inr = loss_fraction x post-onset captured GMV.
    loss_fraction: float


@dataclass(frozen=True, slots=True)
class P1Festival:
    """Festival / sale spike. 5-day bump on volume, all merchants."""

    count: int
    #: Effect size in units of the merchant's OWN daily-count sd. See ConfoundersConfig.
    magnitude_sigma: float
    duration_days: int
    days: list[int]


@dataclass(frozen=True, slots=True)
class P2Outage:
    """Gateway outage. Auth failures up, count down, for a few hours."""

    count: int
    magnitude_sigma: float
    duration_hours: int
    #: Multiplier on the day's intensity during the outage.
    intensity_multiple: float
    days: list[int]


@dataclass(frozen=True, slots=True)
class P3FeeChange:
    """Permanent step: instrument mix shifts toward the cheaper rail."""

    day: int
    magnitude_sigma: float
    #: The rail that gains share.
    target_instrument: str
    #: The step is permanent, but the gate needs a bounded window to score. Long enough
    #: for the 7-day rolling mix feature to register the step.
    window_days: int


@dataclass(frozen=True, slots=True)
class P4NewMethod:
    """A new instrument gains share over ``ramp_days`` on an S-curve."""

    day: int
    ramp_days: int
    magnitude_sigma: float
    target_instrument: str


@dataclass(frozen=True, slots=True)
class P5Regulatory:
    """Tokenisation mandate: permanent CNP/BIN step plus a transient failure elevation."""

    day: int
    transient_days: int
    magnitude_sigma: float


@dataclass(frozen=True, slots=True)
class P6Macro:
    """Always-on sinusoidal modulation of all volume."""

    #: In sd units, not relative units — see ConfoundersConfig for why.
    amplitude_sigma: float
    period_days: int
    #: The window the gate checks is the peak +/- period/peak_window_divisor days.
    peak_window_divisor: float


@dataclass(frozen=True, slots=True)
class ConfoundersConfig:
    """Platform-wide events (P1-P6). 08-generator-v2-spec.md §4 — the v2 contribution.

    **Magnitudes are in sigma units, not relative units.** The spec table says "up 2-4σ",
    and T-114's ``Done when`` clause asks for population mean |z| > 1.0 inside every
    window. With daily counts at Fano 12.25 a merchant's own daily sd is
    ``sqrt(12.25 * lambda)``, so a *relative* multiplier of, say, 1.15 lands at
    |z| ~ 0.2 and could never satisfy the gate. Every magnitude here is therefore
    converted to a multiplier per merchant as
    ``1 + magnitude_sigma * sqrt(target_fano / lambda)``, which reads the parameter the
    way the spec's own table states it.
    """

    enabled: bool
    #: 08-generator-v2-spec.md §4: P1 must hit L2 harder than L4, or the cohort residual
    #: has an unfairly easy job and G5 passes for the wrong reason.
    persona_sensitivity: dict[PersonaId, float]
    P1_festival: P1Festival
    P2_outage: P2Outage
    P3_fee_change: P3FeeChange
    P4_new_method: P4NewMethod
    P5_regulatory: P5Regulatory
    P6_macro: P6Macro


@dataclass(frozen=True, slots=True)
class LabelsConfig:
    #: Uniform(lo, hi) days from label_event_at to label_available_at.
    dispute_delay_days: list[float]
    #: Exponential mean, days from drift_onset_at to label_event_at.
    fraud_to_dispute_mean_days: float
    #: Fraud that is never disputed. Emitted as label=0, not censored — this is the
    #: weak-supervision noise a model that trusts its labels will overfit.
    unreported_rate: float
    #: Chargebacks on merchants that did nothing wrong.
    spurious_chargeback_rate: float


@dataclass(frozen=True, slots=True)
class SettlementConfig:
    cycle_days: int
    #: Merchants below this balance do not bother requesting a payout.
    min_payout_inr: float
    #: P(accelerated settlement) per unit of payout urgency above 1. R1/R2 raise urgency
    #: after onset, and asking for the money faster is half of what bust-out looks like
    #: from the settlement side.
    accelerated_prob_per_urgency: float


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    #: K. Charter §10.4 — load-bearing: a wrong K changes the ranking of the rungs.
    analyst_reviews_per_day: int
    #: K is quoted per this many merchants and scales with the population.
    per_n_merchants: int


@dataclass(frozen=True, slots=True)
class CostsConfig:
    #: v1 measured the asymmetry across three orders of magnitude. `make report` sweeps
    #: these rather than trusting them; a ranking stable across the sweep is the claim.
    fraud_loss_multiplier: float
    false_hold_cost_inr: float
    review_cost_inr: float
    #: Floor on true_loss_amount_inr. schemas.GroundTruth requires it strictly positive
    #: (it is the knapsack weight), and a vanished R1 with almost no post-onset GMV would
    #: otherwise round to zero and fail construction.
    min_true_loss_inr: float


@dataclass(frozen=True, slots=True)
class MarksConfig:
    """Per-transaction mark constants. Everything the engine would otherwise hard-code.

    These read as trivia until one of them is wrong: ``decline_codes`` is the alphabet
    ``f_decline_entropy`` measures, and a single-code list makes that feature identically
    zero for the whole population without failing anything.
    """

    #: Values that a "round amount" snaps to (t_round_amount_share).
    round_amount_values: list[float]
    #: Upper bound of a card-testing probe amount (t_micro_share reads <= 10).
    micro_amount_max: float
    #: Issuer BINs a legitimate merchant's cards are spread over.
    bin_pool_global: int
    device_pool_per_merchant: int
    ip_pool_per_merchant: int
    #: The decline-code alphabet. f_decline_entropy is Shannon entropy over these.
    decline_codes: list[str]
    #: Share of transactions left PENDING at write time.
    pending_rate: float
    #: A refund is Uniform(this, 1.0) x the original. Never above 1.0 — a refund larger
    #: than its capture is property-tested as impossible.
    refund_min_fraction: float
    #: Payer ids above this are freshly minted; below it they belong to a merchant or
    #: ring pool. Keeps the two namespaces from colliding as the run grows.
    payer_id_space: int
    #: R7 mule rings. Members share a payer and device pool, which is the only thing that
    #: makes them jointly detectable and individually unremarkable.
    ring_count: int
    ring_payer_pool: int
    ring_device_pool: int
    #: Jitter on L5's evenly spaced arrivals, in seconds.
    regular_jitter_seconds: float


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    seed: int
    population: PopulationConfig
    arrivals: ArrivalsConfig
    personas: dict[PersonaId, PersonaParams]
    typologies: dict[TypologyId, TypologyParams]
    confounders: ConfoundersConfig
    labels: LabelsConfig
    settlement: SettlementConfig
    capacity: CapacityConfig
    costs: CostsConfig
    marks: MarksConfig
    #: MCC codes drawn per mcc_group. Keys are group names.
    mcc_groups: dict[str, list[str]]
    #: The group no merchant declares and that R4/R9 drift toward. Named here rather than
    #: inferred from the group name, so renaming a group in the YAML cannot silently
    #: make every merchant declarable into it.
    mcc_drift_group: str

    def __post_init__(self) -> None:
        _check_shares(
            {k.value: v.share for k, v in self.personas.items()},
            "personas",
            "share",
        )
        _check_shares(
            {k.value: v.mix for k, v in self.typologies.items()},
            "typologies",
            "mix",
        )
        for pid, p in self.personas.items():
            _check_shares(p.instrument_mix, f"personas.{pid.value}.instrument_mix", None)
            if len(p.hour_weights) != 24:
                raise ConfigError(
                    f"personas.{pid.value}.hour_weights must have 24 entries "
                    f"(one per hour); got {len(p.hour_weights)}"
                )
            if min(p.hour_weights) < 0:
                raise ConfigError(
                    f"personas.{pid.value}.hour_weights must be non-negative; got "
                    f"{min(p.hour_weights)}"
                )
            if p.shape not in _SHAPES:
                raise ConfigError(
                    f"personas.{pid.value}.shape must be one of {sorted(_SHAPES)}; "
                    f"got {p.shape!r}"
                )
        missing_personas = set(PersonaId) - set(self.personas)
        if missing_personas:
            raise ConfigError(
                "personas must define every one of L1-L8 (each is a named hard negative "
                "in 08-generator-v2-spec.md §2); missing "
                f"{sorted(p.value for p in missing_personas)}"
            )
        missing_typologies = set(TypologyId) - set(self.typologies)
        if missing_typologies:
            raise ConfigError(
                "typologies must define every one of R1-R9 (per-typology recall is a "
                "required output, so a missing typology silently removes a reported row); "
                f"missing {sorted(t.value for t in missing_typologies)}"
            )
        missing_sens = set(PersonaId) - set(self.confounders.persona_sensitivity)
        if missing_sens:
            raise ConfigError(
                "confounders.persona_sensitivity must cover every persona; missing "
                f"{sorted(p.value for p in missing_sens)}"
            )
        if not 0.0 <= self.population.prevalence < 1.0:
            raise ConfigError(
                f"population.prevalence must be in [0,1); got {self.population.prevalence}. "
                "prevalence=0 is legal and is exactly what gate G5 runs."
            )
        if self.arrivals.target_fano < 1.0:
            raise ConfigError(
                "arrivals.target_fano must be >= 1.0 — Fano < 1 is under-dispersed and "
                "the negative-binomial has no such parameterisation; got "
                f"{self.arrivals.target_fano}"
            )
        if not 0.0 <= self.arrivals.hawkes_excitation < 1.0:
            raise ConfigError(
                "arrivals.hawkes_excitation is a branching ratio and must be in [0,1) or "
                f"the process is explosive; got {self.arrivals.hawkes_excitation}"
            )
        if len(self.arrivals.dow_factors) != 7:
            raise ConfigError(
                f"arrivals.dow_factors must have 7 entries (Monday first); got "
                f"{len(self.arrivals.dow_factors)}"
            )
        if len(self.labels.dispute_delay_days) != 2:
            raise ConfigError(
                "labels.dispute_delay_days is a [lo, hi] pair of days; got "
                f"{self.labels.dispute_delay_days}"
            )
        if self.labels.dispute_delay_days[0] <= 0:
            raise ConfigError(
                "labels.dispute_delay_days[0] must be > 0: label_available_at is strictly "
                f"after label_event_at (schemas.Label); got {self.labels.dispute_delay_days[0]}"
            )
        if not self.mcc_groups:
            raise ConfigError("mcc_groups must define at least one group")
        if self.mcc_drift_group not in self.mcc_groups:
            raise ConfigError(
                f"mcc_drift_group {self.mcc_drift_group!r} is not one of mcc_groups "
                f"{sorted(self.mcc_groups)}"
            )
        if len(self.mcc_groups) < 2:
            raise ConfigError(
                "mcc_groups needs at least two groups: one is reserved as "
                "mcc_drift_group and merchants declare into the others"
            )
        if not self.marks.decline_codes:
            raise ConfigError(
                "marks.decline_codes must be non-empty — it is the alphabet "
                "f_decline_entropy measures, and an empty one makes that feature "
                "identically zero for the whole population"
            )

    @property
    def analyst_capacity(self) -> int:
        """K, scaled from the quoted per-10k rate to the configured population."""
        return max(
            1,
            round(
                self.capacity.analyst_reviews_per_day
                * self.population.n_merchants
                / self.capacity.per_n_merchants
            ),
        )


#: Persona daily-shape regimes. Implemented in personas.py; named here so a typo in the
#: YAML fails at load rather than silently selecting "flat".
_SHAPES = frozenset({"flat", "seasonal", "growth", "lumpy", "dormant"})


def _check_shares(shares: dict[str, float], where: str, field_name: str | None) -> None:
    total = sum(shares.values())
    if abs(total - 1.0) > SHARE_TOLERANCE:
        suffix = f".{field_name}" if field_name else ""
        raise ConfigError(
            f"{where}{suffix} values must sum to 1.0 +/- {SHARE_TOLERANCE:g}; they sum to "
            f"{total!r} ({total - 1.0:+.3e} off). Values: {shares}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The loader
# ─────────────────────────────────────────────────────────────────────────────

_ENUM_KEYS: dict[type, type[PersonaId] | type[TypologyId] | type[ConfounderId]] = {
    PersonaId: PersonaId,
    TypologyId: TypologyId,
    ConfounderId: ConfounderId,
}


def _build(cls: type[Any], data: Any, path: str) -> Any:
    """Recursively construct ``cls`` from ``data``, naming the dotted path on any error.

    Kept generic rather than writing a ``from_dict`` per dataclass: fourteen hand-written
    constructors is fourteen places for a field to be silently dropped, and the drop is
    invisible because the dataclass still constructs.
    """
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )

    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}

    unknown = set(data) - names
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {sorted(unknown)} for {cls.__name__}. Known keys: "
            f"{sorted(names)}. A renamed parameter still set under its old name is the "
            f"failure this check exists for."
        )
    missing = names - set(data)
    if missing:
        raise ConfigError(
            f"{path}: missing required key(s) {sorted(missing)} for {cls.__name__}. Every "
            f"generator parameter must be stated in the scenario file — there are no "
            f"defaults, because a default is a magic number with a nicer name."
        )

    kwargs = {
        name: _coerce(hints[name], data[name], f"{path}.{name}") for name in names
    }
    try:
        return cls(**kwargs)
    except (ValueError, TypeError) as exc:  # dataclass __post_init__ or a bad type
        raise ConfigError(f"{path}: {exc}") from exc


def _coerce(hint: Any, value: Any, path: str) -> Any:
    origin = get_origin(hint)

    if origin in (Union, UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if value is None:
            return None
        return _coerce(args[0], value, path)

    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _build(hint, value, path)

    if origin is dict:
        key_t, val_t = get_args(hint)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        enum_t = _ENUM_KEYS.get(key_t)
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if enum_t is not None:
                try:
                    key = enum_t(k)
                except ValueError as exc:
                    raise ConfigError(
                        f"{path}: {k!r} is not a valid {enum_t.__name__} "
                        f"({sorted(m.value for m in enum_t)})"
                    ) from exc
            else:
                key = k
            out[key] = _coerce(val_t, v, f"{path}.{k}")
        return out

    if origin is list:
        (item_t,) = get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return [_coerce(item_t, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def load_scenario(path: Path | str) -> ScenarioConfig:
    """Load and validate ``configs/scenario_v2.yaml`` (or any scenario file).

    Raises ``ConfigError`` naming the offending dotted field path. Nothing here is
    stochastic; the seed is carried as data and handed to ``np.random.default_rng`` by
    the caller, so this function is safe to call from a test that must not consume RNG.
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"scenario file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"{p}: file is empty")
    result: ScenarioConfig = _build(ScenarioConfig, raw, p.name)
    return result
