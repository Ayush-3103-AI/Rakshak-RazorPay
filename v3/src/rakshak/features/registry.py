"""The feature register, as code — and the place the state budget is enforced.

07-feature-register.md is the prose version of this module. A feature in the document that
is not here is untestable; a feature here that is not in the document is undocumented. They
are meant to be diffed against each other.

**The budget check runs at import.** If the declared ``state_bytes`` across the registered
features exceeds NFR-04's 4 KB, importing ``rakshak.features`` raises. Failing at startup
is the whole point: a state budget discovered at scale is a state budget discovered by an
outage.
"""

from __future__ import annotations

from rakshak.features.spec import FeatureSpec
from rakshak.features.state import STATE_BYTES_BUDGET
from rakshak.schemas import Tier

__all__ = [
    "ORDER",
    "REGISTRY",
    "StateBudgetExceeded",
    "declared_state_bytes",
    "get",
    "of_tier",
    "register",
    "reset_for_testing",
]


class StateBudgetExceeded(RuntimeError):
    """Raised at import when the declared per-merchant state exceeds NFR-04."""


#: name -> the single shared instance. Feature instances are stateless; the mutable part
#: lives in MerchantState.feature_states.
REGISTRY: dict[str, FeatureSpec] = {}

#: Column order for every FeatureVector. **This is part of the contract** (09-interfaces.md
#: §9): a model trained on one order and scored on another fails silently, and silent is
#: the worst failure mode available here. Registration order is insertion order, so the
#: tier modules are imported in a fixed sequence and features are declared in register
#: order within each.
ORDER: tuple[str, ...] = ()


def declared_state_bytes() -> int:
    """Sum of every registered feature's declared budget. Compared against NFR-04."""
    return sum(spec.state_bytes for spec in REGISTRY.values())


def register(spec_cls: type[FeatureSpec]) -> type[FeatureSpec]:
    """Class decorator: add a feature to the register and re-check the budget.

    Used as::

        @register
        class DailyTxnCount(FeatureSpec):
            name = "v_txn_count"
            ...
    """
    global ORDER
    spec = spec_cls()
    if spec.name in REGISTRY:
        raise ValueError(
            f"feature {spec.name!r} is already registered by "
            f"{type(REGISTRY[spec.name]).__name__}. Names are the join key between this "
            f"registry, 07-feature-register.md, and every trained model's column order — "
            f"two features cannot share one."
        )
    REGISTRY[spec.name] = spec
    ORDER = (*ORDER, spec.name)

    total = declared_state_bytes()
    if total > STATE_BYTES_BUDGET:
        # Unwind before raising, so a caught exception does not leave a half-registered
        # registry that the next import treats as valid.
        del REGISTRY[spec.name]
        ORDER = ORDER[:-1]
        raise StateBudgetExceeded(
            f"registering {spec.name!r} ({spec.state_bytes} B) would take the declared "
            f"per-merchant state to {total} B, over the NFR-04 budget of "
            f"{STATE_BYTES_BUDGET} B. Either shrink a feature's state or cut one — do not "
            f"raise the budget, it is what makes the servability claim checkable."
        )
    return spec_cls


def get(name: str) -> FeatureSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no registered feature named {name!r}; known: {sorted(REGISTRY)}"
        ) from None


def of_tier(tier: Tier) -> tuple[FeatureSpec, ...]:
    """Features in a cascade stage, in ``ORDER``. Stage 0 runs ``of_tier(Tier.T1)`` on
    every merchant every day; stage 1 adds T2 for the top 10%."""
    return tuple(REGISTRY[name] for name in ORDER if REGISTRY[name].tier is tier)


def reset_for_testing() -> None:
    """Empty the registry.

    Registration is import-time and global, which is right for production and awkward for
    a test that wants to register a throwaway feature. This exists for those tests and is
    named so that its appearance in ``src/`` would be obvious in review.
    """
    global ORDER
    REGISTRY.clear()
    ORDER = ()
