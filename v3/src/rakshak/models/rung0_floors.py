"""Rung 0 - the four floors, as rungs rather than as a footnote.

``eval/metrics.floors_at_capacity`` already computes all four numbers and puts them on
every ``EvalResult`` row (FR-021). This module is the other half of the same idea: the
floors as **scorers**, so that ``random_at_k`` and ``volume_rank`` can be run through the
identical decision layer and metric suite as Rungs 1-4 and produce their own complete
rows. A floor that is only ever a column is a floor nobody can inspect.

The construction is deliberately the same one ``savings_of_ranking`` uses - top-K by
score per day, REVIEW on the selected rows - so a Rung-0 row's ``savings`` and the
matching ``savings_floor_*`` column on a Rung-2 row are the same arithmetic on the same
rows. If they ever disagree, something is wrong with the harness and not with the floor.

**``all_hold`` cannot be a row, and that is a fact about the harness, not an omission.**
It alerts on every merchant every day, so ``alerts_per_day`` is the population size and
``build_eval_result`` refuses - correctly - to compute metrics above capacity K. Its
savings is reported on every row as ``savings_floor_all_hold``. See docs/logbook/T-140.md.

Prime Directive 3: nothing here reads a label or a truth field. ``volume`` is the
merchant's own observed captured GMV over the scored window - an event-stream quantity -
and it is passed in, so the same vector feeds both this scorer and the harness's
``volume_rank`` column.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from rakshak.eval.metrics import top_k_by_day
from rakshak.schemas import Action

__all__ = [
    "RANKING_FLOORS",
    "ROW_FLOORS",
    "all_pass_actions",
    "floor_actions",
    "random_scores",
    "rank_normalise",
    "volume_scores",
]

#: The floors that are rankers, and can therefore be scored under capacity K.
RANKING_FLOORS: Final = ("random_at_k", "volume_rank")

#: The floors that produce a complete ``EvalResult`` row. ``all_hold`` is absent by
#: construction, not by choice - see the module docstring.
ROW_FLOORS: Final = ("all_pass", *RANKING_FLOORS)


def rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map any real vector onto [0, 1] preserving order, so it is a legal ``score``.

    ``RungOutput`` requires a calibrated probability because the cost layer treats it as
    one. A floor has no calibration to offer, so it gets its rank instead: order-identical
    to the raw quantity, which is all a top-K selection reads, and honest about carrying no
    probabilistic content. The floors' savings are therefore computed on exactly the
    ranking the harness would have used, and their ECE is meaningless and reported anyway.
    """
    n = values.size
    if n == 0:
        return values.astype(np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def random_scores(n: int, rng: np.random.Generator) -> np.ndarray:
    """``random_at_k``: a uniform score per merchant-day.

    Threaded ``rng``, never a module-level one - v1's headline finding was that random
    won on savings at 20% prevalence, and a floor that cannot be reproduced exactly is a
    floor that cannot be argued about.
    """
    return np.asarray(rng.random(n), dtype=np.float64)


def volume_scores(volume: np.ndarray) -> np.ndarray:
    """``volume_rank``: the dumbest non-random heuristic - alert on the biggest merchants."""
    return rank_normalise(np.asarray(volume, dtype=np.float64))


def floor_actions(score: np.ndarray, day: np.ndarray, k: int) -> np.ndarray:
    """Top-K per day gets REVIEW, everything else PASSes.

    The same fixed decision layer ``savings_of_ranking`` applies, so the difference between
    a floor and a rung is the score vector and nothing else. That is what makes a
    FLOOR-FAIL attributable to the ranking.
    """
    selected = top_k_by_day(score, day, k)
    chosen: np.ndarray = np.where(selected, Action.REVIEW, Action.PASS)
    return chosen


def all_pass_actions(n: int) -> np.ndarray:
    """``all_pass``: never alert. The cost of doing nothing, and the savings denominator."""
    return np.full(n, Action.PASS, dtype=object)
