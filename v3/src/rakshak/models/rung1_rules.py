"""Rung 1 - the static rule engine. The thing a risk team already has.

This is the incumbent a real deployment would be replacing, and it exists so that "the
model beats the floors" is not confused with "the model beats what you have". Charter
§ note: **the bar is LightGBM, not this** - moving the goalposts back to the rule engine
would be the dishonest move. Rung 1 is a rung, not the target.

Three properties are deliberate:

**The thresholds are fixed and declared here, not fitted.** A rule engine tuned on the
validation split is a model with twelve parameters pretending to be a heuristic, and the
comparison against Rung 2 would then be between two fitted things with wildly different
capacity. These are the numbers a fraud analyst would write down: 3 sigma on a z-scored
drift feature, a fifth of the basket in micro-tickets, a third of it international.

**Each rule is graded, not binary.** A binary rule set over twelve rules emits at most
2^12 distinct scores and in practice about eight, which makes the top-K selection under
capacity K almost entirely a tiebreak on row order rather than on risk. Each rule
contributes ``clip((value - threshold) / scale, 0, 1)``, so the score is continuous and
the ranking is a ranking.

**The score is a weighted fraction in [0, 1], and it is not a probability.** It satisfies
``RungOutput``'s domain requirement and nothing more - Rung 1's reported ECE is expected
to be terrible, and that is the honest reading of a rule engine handed to a cost-aware
decision layer that wants probabilities.

Prime Directive 3: features by name only. Nothing here reads a label, a typology or an
onset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from rakshak.features import registry

__all__ = ["RULES", "Rule", "fired_reasons", "score"]


@dataclass(frozen=True, slots=True)
class Rule:
    """One analyst heuristic over one registered feature.

    ``scale`` is the distance past ``threshold`` at which the rule is fully fired. It is
    the difference between "3.1 sigma" and "31 sigma" mattering, which a binary rule
    throws away.
    """

    feature: str
    threshold: float
    scale: float
    weight: float
    why: str

    def __post_init__(self) -> None:
        if self.scale <= 0.0 or self.weight <= 0.0:
            raise ValueError(f"rule {self.feature!r} needs positive scale and weight")
        if self.feature not in registry.REGISTRY:
            raise KeyError(
                f"rule names {self.feature!r}, which is not a registered feature. The rule "
                "engine reads the same register the model does, or the two are not "
                "comparable."
            )

    def severity(self, values: np.ndarray) -> np.ndarray:
        """0 below the threshold, 1 at ``threshold + scale``, linear between."""
        graded: np.ndarray = np.clip((values - self.threshold) / self.scale, 0.0, 1.0)
        return graded


#: The engine. Twelve rules over the T1/T2 register, weighted by how much a fraud analyst
#: would actually move on each one. Written before any validation number was looked at.
RULES: Final[tuple[Rule, ...]] = (
    Rule("v_gmv_z", 3.0, 3.0, 1.0, "GMV far above this merchant's own norm"),
    Rule("v_txn_count_z", 3.0, 3.0, 0.8, "transaction count far above its own norm"),
    Rule("f_auth_fail_rate_z", 3.0, 3.0, 1.0, "authorisation failures spiking"),
    Rule("d_refund_rate_z", 3.0, 3.0, 0.9, "refund rate spiking"),
    Rule("v_declared_ratio", 4.0, 4.0, 0.8, "processing far above declared volume"),
    Rule("v_dormant_burst", 3.0, 3.0, 0.7, "a dormant merchant woke up abruptly"),
    Rule("t_micro_share", 0.05, 0.20, 0.9, "an unusual share of micro-tickets"),
    Rule("t_round_amount_share", 0.05, 0.20, 0.6, "an unusual share of round amounts"),
    Rule("i_intl_share", 0.30, 0.30, 0.6, "an unusual share of international traffic"),
    Rule("v_gmv_accel", 2.0, 3.0, 0.5, "GMV accelerating, not merely rising"),
    Rule("h_weekend_share_z", 3.0, 3.0, 0.3, "the weekly shape of the business changed"),
    Rule("i_mix_jsd", 0.50, 0.40, 0.4, "the instrument mix diverged from its baseline"),
)

_TOTAL_WEIGHT: Final = sum(rule.weight for rule in RULES)


def _matrix(x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    """Per-rule severity, ``(rows x rules)``."""
    index = {name: i for i, name in enumerate(columns)}
    missing = [rule.feature for rule in RULES if rule.feature not in index]
    if missing:
        raise KeyError(f"the panel is missing features the rule engine needs: {missing}")
    return np.column_stack([rule.severity(x[:, index[rule.feature]]) for rule in RULES])


def score(x: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    """Weighted fraction of the rule set that fired, in [0, 1]. Not a probability."""
    if x.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    weights = np.array([rule.weight for rule in RULES], dtype=np.float64)
    total: np.ndarray = _matrix(x, columns) @ weights / _TOTAL_WEIGHT
    return total


def fired_reasons(
    x: np.ndarray, columns: tuple[str, ...], row: int, top: int = 3
) -> list[str]:
    """The ``top`` rules contributing most to one row's score, as merchant-readable text.

    Rung 1's audit trail. FR-033's ``pred_contrib`` route needs a booster and there is not
    one here, so the contribution is the rule's own weighted severity - which for a linear
    rule engine is the same quantity, exactly.
    """
    index = {name: i for i, name in enumerate(columns)}
    contributions = [
        (rule.weight * float(rule.severity(x[row : row + 1, index[rule.feature]])[0]), rule)
        for rule in RULES
    ]
    contributions.sort(key=lambda pair: -pair[0])
    return [
        registry.get(rule.feature).explain(float(x[row, index[rule.feature]]))
        for value, rule in contributions[:top]
        if value > 0.0
    ]
