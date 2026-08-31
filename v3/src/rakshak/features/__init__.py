"""The feature layer: one definition, two runners, parity asserted in CI.

Importing this package registers every feature and enforces the NFR-04 state budget. It
must never import from ``rakshak.generator`` or touch ``GroundTruth`` — Prime Directive 3,
enforced by an AST scan in ``tests/gates/test_no_ground_truth_import.py``.
"""

# The tier modules are imported here in a FIXED sequence, because import order *is* column
# order (registry.ORDER) and column order is part of the contract (09-interfaces.md §9). A
# model trained on one order and scored on another fails silently. T-120 appended tier1,
# T-122 appends tier2 after it; append deliberately, never alphabetically. ruff's import
# sort happens to agree with the deliberate order here (tier1 < tier2) — if it ever stops
# agreeing, the deliberate order wins and the sort gets the noqa.
from rakshak.features import registry, tier1, tier2
from rakshak.features.spec import PARITY_TOLERANCE, FeatureSpec
from rakshak.features.state import (
    STATE_BYTES_BUDGET,
    BaselineStats,
    FeatureState,
    MerchantState,
)

__all__ = [
    "PARITY_TOLERANCE",
    "STATE_BYTES_BUDGET",
    "BaselineStats",
    "FeatureSpec",
    "FeatureState",
    "MerchantState",
    "registry",
    "tier1",
    "tier2",
]
