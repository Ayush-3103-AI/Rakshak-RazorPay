"""The bounded online state a merchant carries between epochs.

Everything a feature needs to produce today's value from today's event and nothing else
lives here. The 4 KB ceiling (NFR-04) is not a performance preference — it is what makes
the claim "this system is servable" checkable rather than asserted. A feature that cannot
fit is a feature that does not ship, and that trade is made here rather than argued later.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, date, datetime
from typing import Any

import numpy as np

from rakshak.schemas import SCHEMA_VERSION, MerchantProfile

__all__ = [
    "STATE_BYTES_BUDGET",
    "BaselineStats",
    "FeatureState",
    "MerchantState",
    "pack",
    "unpack",
]

#: NFR-04. A serialized MerchantState must fit in this, asserted in tests/perf/.
STATE_BYTES_BUDGET = 4096


# ─────────────────────────────────────────────────────────────────────────────
# The packed wire format (T-150).
#
# Online state used to be measured with ``pickle.dumps``, and pickle charges roughly 120 B
# of framing per small object, 9 B for every float and ~14 B for every date. Across 28
# feature states — most of which hold a ring of daily ``(date, float, float, float)``
# tuples and four of which hold a week of histograms — that framing was the majority of the
# measured size. LIMITATIONS.md §2 recorded it as "a serialization problem, not a feature
# problem", and this is the fix for the serialization half.
#
# One tagged little-endian buffer. Values contiguous, one short class name per object
# instead of a pickle class reference, and a homogeneous list of floats written as a raw
# array rather than element by element. When every float in such a list is a non-negative
# whole number — which every histogram in the feature layer is, because its elements are
# event counts — it narrows to uint32, checked per array against the actual values and
# never assumed from the field's declared type.
#
# It is a real serialization and not an estimator: ``unpack(pack(x)) == x`` for every state
# the layer produces, asserted in tests/perf/test_state_size.py. This is what a Redis-backed
# deployment would actually store, so it is what NFR-04 should be measured against.
#
# ponytail: pure-Python codec, ~6x slower than pickle in both directions (measured: pack
# 1.50 ms / unpack 0.85 ms against pickle's 0.23 / 0.14 on the same state). That is 8.5 s of
# the 30 s NFR-03 sweep spent deserializing, which fits with room but is the largest single
# line in it. It buys 27% off the size, which is the budget that is actually contested. If
# the sweep ever gets tight, the upgrade is to encode the whole state as one numpy structured
# buffer per feature type rather than walking values in Python — not to go back to pickle,
# whose framing is the thing being measured out.
# ─────────────────────────────────────────────────────────────────────────────

_TAG_NONE = 0
_TAG_FALSE = 1
_TAG_TRUE = 2
_TAG_INT = 3
_TAG_FLOAT = 4
_TAG_STR = 5
_TAG_DATE = 6
_TAG_DATETIME = 7
_TAG_LIST = 8
_TAG_TUPLE = 9
_TAG_DICT = 10
_TAG_OBJ = 11
#: A list whose every element is a float, written as a contiguous array.
_TAG_LIST_F8 = 12
#: ...and the same list when every value in it is a non-negative whole number.
_TAG_LIST_U32 = 13

_I8 = struct.Struct("<q")
_F8 = struct.Struct("<d")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")

_U32_MAX = float(2**32 - 1)

#: Classes ``unpack`` is allowed to construct. ``FeatureState.__init_subclass__`` adds each
#: feature's own state type; the three below are added explicitly. An allowlist rather than
#: an import-by-name, because a deserializer that will construct whatever class it is told
#: to is the pickle vulnerability again with extra steps.
_TYPES: dict[str, type] = {}


def _emit(value: object, out: bytearray) -> None:
    # Order matters twice over: `bool` before `int` because it is a subclass of it, and
    # `datetime` before `date` for the same reason.
    if value is None:
        out.append(_TAG_NONE)
    elif value is True:
        out.append(_TAG_TRUE)
    elif value is False:
        out.append(_TAG_FALSE)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        out.append(_TAG_STR)
        out += _U16.pack(len(raw))
        out += raw
    elif isinstance(value, datetime):
        # The repo's timestamps are tz-aware UTC (09-interfaces.md). `datetime` cannot hold
        # nanoseconds at all, so microseconds since the epoch is lossless for anything that
        # fits in one — and the roundtrip test asserts that rather than assuming it.
        out.append(_TAG_DATETIME)
        out += _I8.pack(int(round(value.timestamp() * 1_000_000)))
    elif isinstance(value, date):
        out.append(_TAG_DATE)
        out += _U32.pack(value.toordinal())
    elif isinstance(value, (int, np.integer)):
        out.append(_TAG_INT)
        out += _I8.pack(int(value))
    elif isinstance(value, (float, np.floating)):
        out.append(_TAG_FLOAT)
        out += _F8.pack(float(value))
    elif isinstance(value, list):
        _emit_list(value, out)
    elif isinstance(value, tuple):
        out.append(_TAG_TUPLE)
        out += _U32.pack(len(value))
        for item in value:
            _emit(item, out)
    elif isinstance(value, dict):
        out.append(_TAG_DICT)
        out += _U32.pack(len(value))
        for key, item in value.items():
            _emit(key, out)
            _emit(item, out)
    elif is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _TYPES.get(name) is not type(value):
            raise TypeError(
                f"{name} is not in the packed-state allowlist. A FeatureState subclass "
                f"registers itself on definition; anything else is added to _TYPES in "
                f"state.py by name, deliberately."
            )
        out.append(_TAG_OBJ)
        _emit(name, out)
        for f in fields(value):
            _emit(getattr(value, f.name), out)
    else:
        raise TypeError(
            f"{type(value).__name__} has no packed representation. Online state is "
            f"scalars, dates, lists, tuples and allowlisted dataclasses — a feature "
            f"reaching for anything else is a feature whose state nobody can size (NFR-04)."
        )


def _emit_list(value: list[Any], out: bytearray) -> None:
    if value and all(type(item) is float for item in value):
        arr = np.array(value, dtype=np.float64)
        out += bytes([_TAG_LIST_F8])
        out += _U32.pack(len(value))
        if np.all(arr >= 0.0) and np.all(arr <= _U32_MAX) and np.all(arr == np.floor(arr)):
            out[-5] = _TAG_LIST_U32
            out += arr.astype(np.uint32).tobytes()
        else:
            out += arr.tobytes()
        return
    out.append(_TAG_LIST)
    out += _U32.pack(len(value))
    for item in value:
        _emit(item, out)


def _read(buf: bytes, pos: int) -> tuple[Any, int]:
    tag = buf[pos]
    pos += 1
    if tag == _TAG_NONE:
        return None, pos
    if tag == _TAG_TRUE:
        return True, pos
    if tag == _TAG_FALSE:
        return False, pos
    if tag == _TAG_INT:
        return _I8.unpack_from(buf, pos)[0], pos + 8
    if tag == _TAG_FLOAT:
        return _F8.unpack_from(buf, pos)[0], pos + 8
    if tag == _TAG_STR:
        n = _U16.unpack_from(buf, pos)[0]
        pos += 2
        return buf[pos : pos + n].decode("utf-8"), pos + n
    if tag == _TAG_DATE:
        return date.fromordinal(_U32.unpack_from(buf, pos)[0]), pos + 4
    if tag == _TAG_DATETIME:
        micros = _I8.unpack_from(buf, pos)[0]
        return datetime.fromtimestamp(micros / 1_000_000, tz=UTC), pos + 8
    if tag == _TAG_LIST_F8:
        n = _U32.unpack_from(buf, pos)[0]
        pos += 4
        values: list[float] = np.frombuffer(buf, dtype=np.float64, count=n, offset=pos).tolist()
        return values, pos + 8 * n
    if tag == _TAG_LIST_U32:
        n = _U32.unpack_from(buf, pos)[0]
        pos += 4
        counts = np.frombuffer(buf, dtype=np.uint32, count=n, offset=pos)
        return counts.astype(np.float64).tolist(), pos + 4 * n
    if tag in (_TAG_LIST, _TAG_TUPLE):
        n = _U32.unpack_from(buf, pos)[0]
        pos += 4
        items: list[Any] = []
        for _ in range(n):
            item, pos = _read(buf, pos)
            items.append(item)
        return (items if tag == _TAG_LIST else tuple(items)), pos
    if tag == _TAG_DICT:
        n = _U32.unpack_from(buf, pos)[0]
        pos += 4
        mapping: dict[Any, Any] = {}
        for _ in range(n):
            key, pos = _read(buf, pos)
            item, pos = _read(buf, pos)
            mapping[key] = item
        return mapping, pos
    if tag == _TAG_OBJ:
        name, pos = _read(buf, pos)
        cls = _TYPES.get(name)
        if cls is None:
            raise ValueError(
                f"packed state names {name!r}, which is not in the allowlist. Either the "
                f"buffer predates a rename or it came from somewhere it should not have."
            )
        args: list[Any] = []
        for _ in fields(cls):
            arg, pos = _read(buf, pos)
            args.append(arg)
        return cls(*args), pos
    raise ValueError(f"unknown tag {tag} at offset {pos - 1} in packed state")


def pack(value: object) -> bytes:
    """Serialize online state to the packed wire format. See the module note above."""
    out = bytearray()
    _emit(value, out)
    return bytes(out)


def unpack(data: bytes) -> Any:
    """Inverse of :func:`pack`. Only allowlisted classes are constructed."""
    value, pos = _read(data, 0)
    if pos != len(data):
        raise ValueError(f"packed state has {len(data) - pos} trailing bytes")
    return value


@dataclass(slots=True)
class FeatureState:
    """Base for a feature's own online state.

    Subclasses add their own slotted fields. There is no required interface beyond being a
    dataclass of packable values, because forcing every feature to hand-write a size
    accessor is boilerplate that would be wrong by the third feature; ``nbytes`` measures
    the real thing instead.
    """

    def __init_subclass__(cls) -> None:
        # Registers the state type with the packed codec's allowlist.
        #
        # No `super().__init_subclass__()` call, and that is not an oversight:
        # `@dataclass(slots=True)` does not mutate the class it decorates, it *replaces* it
        # with a fresh one, and the zero-argument `super()` in a method of the discarded
        # original still closes over the discarded original — so the call raises
        # "obj must be an instance or subtype of type" on the first subclass. There is no
        # cooperative chain above `object` here to keep alive, so the call goes.
        #
        # The replacement re-enters this hook, which is what makes the entry that survives
        # the slotted class callers actually hold — the one `unpack` has to construct.
        _TYPES[cls.__name__] = cls

    def nbytes(self) -> int:
        """Actual packed size. The registry checks *declared* ``state_bytes`` at import so
        the budget fails at startup; this measures what the declaration promised, so a
        feature that quietly outgrows its declaration is caught in tests/perf/."""
        return len(pack(self))


@dataclass(slots=True)
class BaselineStats:
    """Welford accumulators over the merchant's own post-onboarding warmup window.

    **Frozen after ``warmup_days``, deliberately.** A rolling baseline lets a slow-ramp
    adversary walk the baseline along with it — the merchant is always "normal relative to
    last month" while last month keeps getting worse. That is precisely how typology R2
    defeats naive drift detection, so the baseline stops moving and the z-scores are taken
    against a fixed reference. Name this choice in the writeup; it is a design decision,
    not an implementation shortcut.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    frozen: bool = False

    def observe(self, x: float) -> None:
        if self.frozen:
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def freeze(self) -> None:
        self.frozen = True

    @property
    def variance(self) -> float:
        # Population variance over the warmup sample. n<2 has no dispersion to speak of.
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(self.variance**0.5)

    def z(self, x: float, *, floor: float = 1e-9) -> float:
        """z-score of ``x`` against the frozen baseline.

        ``floor`` keeps a merchant whose warmup window was perfectly flat from producing
        an infinite z on its first varied day. Without it, a dormant merchant's first
        transaction is an alert every time, which is a false-positive generator, not a
        detector.
        """
        if self.count == 0:
            return 0.0
        return (x - self.mean) / max(self.std, floor)


@dataclass(slots=True)
class MerchantState:
    """The whole of what the online path remembers about one merchant.

    Serialized size <= ``STATE_BYTES_BUDGET`` (NFR-04). The registry sums each feature's
    *declared* ``state_bytes`` at import time and refuses to load if the total exceeds the
    budget — fail at startup, not at scale.
    """

    merchant_id: str
    profile: MerchantProfile
    baseline: BaselineStats = field(default_factory=BaselineStats)
    feature_states: dict[str, FeatureState] = field(default_factory=dict)
    last_event_time: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def warmup_elapsed(self, as_of: datetime, warmup_days: int) -> bool:
        return (as_of - self.profile.onboarded_at).days >= warmup_days

    def maybe_freeze_baseline(self, as_of: datetime, warmup_days: int) -> None:
        if not self.baseline.frozen and self.warmup_elapsed(as_of, warmup_days):
            self.baseline.freeze()

    def pack(self) -> bytes:
        """The bytes a store would hold for this merchant. NFR-04 is measured on these."""
        return pack(self)

    @staticmethod
    def unpack(data: bytes) -> MerchantState:
        state = unpack(data)
        if not isinstance(state, MerchantState):
            raise TypeError(f"packed buffer holds a {type(state).__name__}, not a MerchantState")
        return state

    def nbytes(self) -> int:
        return len(pack(self))


# The three non-FeatureState classes the packed format has to construct. Listed by hand,
# after their definitions, so the allowlist is a decision rather than a side effect.
_TYPES["BaselineStats"] = BaselineStats
_TYPES["MerchantState"] = MerchantState
_TYPES["MerchantProfile"] = MerchantProfile


def day_of(when: datetime | date) -> date:
    """The epoch a timestamp belongs to. Daily epochs, defined once so that a feature
    cannot accidentally bucket by local date on one path and UTC date on the other —
    which is a parity failure that only appears for events near midnight."""
    return when.date() if isinstance(when, datetime) else when
