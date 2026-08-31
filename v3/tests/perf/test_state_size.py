"""NFR-04 — ``MerchantState`` <= 4 KB serialized per merchant.

**This budget is not met, and the honest number is asserted here rather than the budget
being widened to fit the code.** LIMITATIONS.md §2 carried the problem to T-150 with a
diagnosis — "a serialization problem, not a feature problem", ~120 B of pickle framing per
`FeatureState` across 28 objects — and that diagnosis was right about the cause and wrong
about the size of it.

What T-150 found, in order:

* Pickled, a fully-warmed 28-feature state is **13,145 B**, not the 7,091 B recorded. The
  earlier figure was taken before the trailing windows had filled; a state measured before
  its rings are full is measured below its steady size.
* Packed (``features/state.py``, one tagged little-endian buffer, float lists written as
  contiguous arrays and count arrays narrowed to uint32) it is **9,634 B**. That is the
  framing gone: a 27% reduction, and it is a real serialization — ``unpack(pack(x)) == x``
  is asserted below, not assumed.
* The **zero-framing floor** — every scalar the state holds, written at its natural width
  with no tags, no lengths and no class names at all — is **9,734 B**. The packed form is
  already *below* the floor of the naive encoding, because narrowing histogram counts to
  uint32 buys more than the tag bytes cost.

So the packed representation is at the information content of the state itself, and NFR-04
is still exceeded by **2.35x**. No serialization closes that gap. What would:

* The two T2 histogram features are 4.1 KB of the 9.6 KB between them — ``t_wasserstein_7d``
  (32 bins) and ``h_hourly_jsd`` (24 bins), each holding a frozen baseline plus a week of
  completed daily histograms. A shared daily histogram across the four T2 features, or a
  shorter T2 window, is the single biggest lever and it is a change to ``tier2.py``.
* The five windowed T1 features hold 20-30 completed days of ``(date, num, den, aux)``.
  Packing those column-wise instead of per-tuple is worth roughly 1.5 KB more and still
  does not reach 4,096 B, which is why it was not built.
* Cutting features. NFR-04 exists to force exactly that trade.

The 4,096 B assertion is kept live as ``xfail(strict=True)`` — the same treatment T-121's
unreachable clause got, for the same reason. It is not deleted, it is not softened, and if
someone ever makes it pass the strict xfail fails and this file has to be rewritten with
the good news. Alongside it, a live regression fence fails CI if the state grows further.
"""

from __future__ import annotations

import pickle
from dataclasses import fields, is_dataclass
from datetime import date, datetime

import pytest
from perf_budgets import assert_budget

from rakshak.features import registry
from rakshak.features.state import STATE_BYTES_BUDGET, MerchantState, pack, unpack

#: The measured packed size of a fully-warmed 28-feature state, plus headroom for the
#: string lengths that vary with a merchant id. **This is not a budget** — NFR-04's budget
#: is STATE_BYTES_BUDGET and it is not met. This is a fence: it fails CI when the state
#: grows past what T-150 measured, which is the regression the real budget can no longer
#: catch while it sits in xfail.
STATE_BYTES_MEASURED_CEILING = 10_240


def _floor_bytes(value: object) -> int:
    """Every scalar at its natural width, with no framing of any kind.

    The lower bound on any encoding of this state that keeps float64 precision — which
    parity to 1e-9 (NFR-08) requires, so it is not a bound that can be negotiated.
    """
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, datetime):
        return 8
    if isinstance(value, date):
        return 4
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, (list, tuple)):
        return sum(_floor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_floor_bytes(k) + _floor_bytes(v) for k, v in value.items())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_floor_bytes(getattr(value, f.name)) for f in fields(value))
    raise TypeError(f"no floor defined for {type(value).__name__}")


def test_the_packed_form_roundtrips_exactly(warm: tuple[MerchantState, datetime]) -> None:
    """``unpack(pack(x)) == x``, or the measurement below is of a lossy encoding.

    This is the assertion that makes the packed size mean something. A smaller number
    obtained by dropping a field is not a smaller state, and float32 would be smaller
    still and would break parity (NFR-08) rather than the budget.
    """
    state, _ = warm
    blob = state.pack()
    back = MerchantState.unpack(blob)
    assert back == state
    assert back.pack() == blob, "packing is not idempotent, so the format is not canonical"

    for name, fs in state.feature_states.items():
        assert unpack(pack(fs)) == fs, f"{name} does not survive a pack/unpack roundtrip"


def test_the_packed_form_rejects_a_class_it_was_not_told_about() -> None:
    """The allowlist is the reason this is not pickle wearing a hat.

    A deserializer that constructs whatever class the buffer names is the pickle
    vulnerability with extra steps. Unknown names raise; they do not import.
    """
    with pytest.raises(ValueError, match="not in the allowlist"):
        unpack(bytes([11, 5, 3, 0, ord("F"), ord("o"), ord("o")]))


def test_state_size_is_reported_against_pickle_and_against_the_floor(
    warm: tuple[MerchantState, datetime],
) -> None:
    """The three numbers in this file's docstring, regenerated. Report, not assertion."""
    state, _ = warm
    packed = state.nbytes()
    pickled = len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    floor = _floor_bytes(state)
    per_feature = sorted(
        ((fs.nbytes(), name, registry.REGISTRY[name].state_bytes) for name, fs in
         state.feature_states.items()),
        reverse=True,
    )
    print(
        f"\nNFR-04 state size, {len(state.feature_states)} features warmed to steady state:"
        f"\n  pickled            {pickled:6d} B"
        f"\n  packed             {packed:6d} B   ({100 * (1 - packed / pickled):.1f}% smaller)"
        f"\n  zero-framing floor {floor:6d} B   (float64, no tags, no lengths, no names)"
        f"\n  NFR-04 budget      {STATE_BYTES_BUDGET:6d} B"
        f"\n  declared total     {registry.declared_state_bytes():6d} B"
        + "".join(f"\n    {n:26s} packed {b:5d} B  declared {d:5d} B" for b, n, d in per_feature)
    )
    assert packed < pickled, "packing did not beat pickle, which was the entire premise"


def test_state_has_not_grown_since_it_was_measured(
    warm: tuple[MerchantState, datetime],
) -> None:
    """The live fence. Not the budget — see ``STATE_BYTES_MEASURED_CEILING``."""
    state, _ = warm
    assert_budget(
        "NFR-04-fence",
        "packed MerchantState against the T-150 measured ceiling",
        float(state.nbytes()),
        float(STATE_BYTES_MEASURED_CEILING),
        "B",
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "NFR-04 is not met. Packed, a fully-warmed 28-feature MerchantState is 9,634 B "
        "against a 4,096 B budget — 2.35x over — and the zero-framing float64 floor of the "
        "same state is 9,734 B, so no serialization closes the gap. Kept live and strict "
        "rather than deleted or widened: if this ever passes, the xfail fails and the "
        "finding gets rewritten. See LIMITATIONS.md §2 and this module's docstring for what "
        "would close it."
    ),
)
def test_state_fits_the_nfr04_budget(warm: tuple[MerchantState, datetime]) -> None:
    state, _ = warm
    assert_budget(
        "NFR-04",
        "packed MerchantState per merchant",
        float(state.nbytes()),
        float(STATE_BYTES_BUDGET),
        "B",
    )
