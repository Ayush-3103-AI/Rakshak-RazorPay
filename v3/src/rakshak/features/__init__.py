"""The feature layer: one definition, two runners, parity asserted in CI.

Importing this package registers every feature and enforces the NFR-04 state budget. It
must never import from ``rakshak.generator`` or touch ``GroundTruth`` — Prime Directive 3,
enforced by an AST scan in ``tests/gates/test_no_ground_truth_import.py``.
"""

from rakshak.features import registry
from rakshak.features.spec import PARITY_TOLERANCE, FeatureSpec
from rakshak.features.state import (
    STATE_BYTES_BUDGET,
    BaselineStats,
    FeatureState,
    MerchantState,
)

# Tier modules are imported here, in this order, because import order *is* column order
# (registry.ORDER) and column order is part of the contract. T-120 appends tier1, T-122
# appends tier2. Adding an import here changes every trained model's feature layout, so it
# is done deliberately and never alphabetically.

__all__ = [
    "PARITY_TOLERANCE",
    "STATE_BYTES_BUDGET",
    "BaselineStats",
    "FeatureSpec",
    "FeatureState",
    "MerchantState",
    "registry",
]
