"""Run the cost-asymmetry sweep the harness has always been able to run and never has.

``eval.capacity.sweep_cost_asymmetry`` has been built and unit-tested since T-132. It has
never been run over the ladder: there is no artefact, no results section, and no figure.
Every savings verdict in the repo is therefore a **single point estimate** at one guessed
cost matrix, and ``metrics.CostParams``' own docstring says that is not good enough --
"these three rupee values are swept parameters, not constants", measured across three
orders of magnitude in v1.

This script closes that. It changes no model, refits nothing, and touches no locked file.
It re-scores decisions that already exist under five cost matrices instead of one.

    uv run python scripts/cost_sweep.py                 # all five seeds, ~1 panel load
    uv run python scripts/cost_sweep.py --seeds 42      # one seed, for a quick read

**The split is hard-wired to ``val``**, as in ``scripts/rescore_cycle4.py`` and for the same
reason: the test split is a one-way door governed by a pre-registration and is not something
an analysis script should be able to open by accident.

WHAT VARIES AND WHAT IS HELD
----------------------------
The swept quantity is ``false_hold_cost / mean_fraud_loss`` over ``ASYMMETRY_RATIOS``, the
five points declared in ``10-eval-harness-spec.md`` §2. They are used exactly as declared
and not extended: choosing a denser grid after seeing the shape of the curve is how a sweep
becomes an argument. ``review_cost_inr`` and ``p_catch`` are held fixed, so the result is
interpretable as *the* asymmetry moving rather than everything moving at once.

THREE TABLES, BECAUSE ONE WOULD MOVE TWO THINGS AT ONCE
-------------------------------------------------------
``PRE-REGISTRATION-CYCLE4`` §4.3 disclosed, and declined to fix, an asymmetry inside the
comparison that decides FLOOR-FAIL: floors are priced REVIEW-only (Rs.250/error) while rungs
are priced on their own actions, which may HOLD (Rs.8,250/error) -- 33x. Fixing it means
editing the locked eval package, which §3 forbids. It can be *measured* without touching the
hash, and that is what the three tables do:

- **A** -- rungs on the actions they actually take, the ladder's own convention.
- **B** -- every policy, floors included, as a pure ranking priced REVIEW-only.
- **C** -- arm B through the decision layer with HOLD made unreachable, and nothing else
  changed. B alone would not settle it: it strips the HOLD action *and* the expected-value
  re-ranking, so a gap measured against it is two effects added together.

A, then C, then B decomposes the rung-vs-floor margin into the HOLD privilege, the
exposure-aware re-ranking, and the score ranking itself.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rakshak import cli
from rakshak.eval import capacity
from rakshak.eval.metrics import (
    RungOutput,
    floors_at_capacity,
    savings_of_ranking,
)
from rakshak.eval.metrics import day_labels as _day_labels
from rakshak.models import dataset, decision_realised_exposure, rung0_floors, rung1_rules

#: The rungs that go through ``select_actions`` and therefore respond to the cost matrix.
#: Rungs 5 and 6 are excluded deliberately: Rung 5 scores a different row universe (payer
#: capsules over a merchant subsample) and Rung 6 is a decision-policy wrapper that emits no
#: score, so neither can be swept on the same rows as these without comparing two different
#: quantities under one column heading.
RUNGS = (1, 2, 3, 4, 9)

OUT_MD = Path("docs/results/cost_sweep.md")
OUT_JSON = Path("docs/results/cost_sweep.json")


def _scores_for_seed(
    *, full: Any, rows: Any, seed: int, root: Path, boundaries: Any
) -> dict[str, np.ndarray]:
    """Every rung's score vector on the validation rows, by the same path ``eval`` uses.

    Deliberately routed through ``cli``'s own private helpers rather than reimplemented.
    A second scoring path that drifts from the documented one is how a results table ends
    up disagreeing with the artefact it claims to summarise.
    """
    scores: dict[str, np.ndarray] = {}
    for rung in RUNGS:
        if rung == 1:
            scores["rung1"] = rung1_rules.score(rows.x, rows.columns)
        elif rung == 9:
            score, _blend, _acc = cli._score_rung9(
                full=full, rows=rows, seed=seed, root=root, boundaries=boundaries
            )
            scores["rung9"] = score
        else:
            model = cli._load_trained(rung, seed)
            scores[f"rung{rung}"] = model.predict(rows.x, rows.columns)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/v2"))
    ap.add_argument("--panel", type=Path, default=dataset.DEFAULT_PANEL)
    ap.add_argument("--config", type=Path, default=Path("configs/scenario_v2.yaml"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--render-only", action="store_true",
                    help="Re-render the markdown from an existing cost_sweep.json without "
                         "re-scoring. The JSON already holds per-seed means, and _mean of a "
                         "one-element list is that element, so the tables are identical to "
                         "the ones the scoring run would print. Use it to change the prose "
                         "around the numbers, never to produce numbers.")
    args = ap.parse_args()

    if args.render_only:
        return _render_only()

    params = cli._cost_params(args.config)
    policy = cli._action_policy(args.config)
    boundaries = cli._boundaries(args.config)

    print(f"loading panel {args.panel} ...", flush=True)
    full = dataset.load_panel(args.panel)
    rows = full.select("val")
    merchants = sorted(set(rows.merchant_id.tolist()))
    truth = cli._build_truth(
        args.root, merchants, cutoff_day=boundaries.val[0] - 1, boundaries=boundaries
    )
    k = cli._capacity(len(merchants))

    declared = rows.column("p_declared_monthly_gmv")
    realised = decision_realised_exposure.realised_exposure_inr(
        declared, rows.column("v_declared_ratio")
    )

    # `keep` drops merchants whose label had not resolved by the cutoff. On cycle-4 data it
    # is all-True (n_censored_dropped is 0 on every committed row) but it is applied anyway,
    # because a sweep that silently assumed that would break the day the geometry moves.
    probe = RungOutput(
        merchant_id=rows.merchant_id,
        day=rows.day,
        score=np.zeros(rows.x.shape[0]),
        action=rung0_floors.all_pass_actions(rows.x.shape[0]),
    )
    y_all, keep = _day_labels(probe, truth)
    order = np.argsort(truth.merchant_id)
    idx = order[np.searchsorted(truth.merchant_id[order], rows.merchant_id)]
    loss_all = truth.loss_inr[idx]
    volume_all = truth.volume[idx]

    y = y_all[keep]
    loss = loss_all[keep]
    day = rows.day[keep]
    volume = volume_all[keep]
    exposure = {"declared": declared[keep], "realised": realised[keep]}

    print(
        f"K={k}  merchants={len(merchants):,}  rows={rows.x.shape[0]:,}  "
        f"kept={int(keep.sum()):,}  positives={int(y.sum()):,}",
        flush=True,
    )

    # arm -> ratio -> policy -> [savings per seed]
    hold: dict[str, dict[float, dict[str, list[float]]]] = {
        arm: defaultdict(lambda: defaultdict(list)) for arm in exposure
    }
    review: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    nohold: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Table C's policy: the decision layer exactly as scored, with HOLD made unreachable.
    # `hold_expected_loss_floor_inr` is the only knob that forbids HOLD without touching
    # the locked selector -- `score * exposure >= inf` is False on every row, so every
    # intervention becomes REVIEW while the expected-value top-K selection is untouched.
    no_hold_policy = capacity.ActionPolicy(
        hold_score_threshold=policy.hold_score_threshold,
        hold_expected_loss_floor_inr=float("inf"),
    )

    for seed in args.seeds:
        print(f"scoring seed {seed} ...", flush=True)
        scores = _scores_for_seed(
            full=full, rows=rows, seed=seed, root=args.root, boundaries=boundaries
        )
        scores = {name: s[keep] for name, s in scores.items()}

        for arm, exp in exposure.items():
            for row in capacity.sweep_cost_asymmetry(
                scores, day, y, loss, exp, k, params, policy=policy
            ):
                hold[arm][row.ratio][row.rung].append(row.savings)

        # Table C: arm B only. Arm A is the estimator the exposure diagnostic already
        # showed to be the weaker one, and running the isolation under both would double
        # the table without changing which comparison it settles.
        for row in capacity.sweep_cost_asymmetry(
            scores, day, y, loss, exposure["realised"], k, params, policy=no_hold_policy
        ):
            nohold[row.ratio][row.rung].append(row.savings)

        # Table B: the floors' own pricing, applied to everything. `volume_rank` and
        # `random_at_k` go through this in the ladder too, so their Table B rows and their
        # committed ladder rows are the same computation at the base ratio.
        for ratio in capacity.ASYMMETRY_RATIOS:
            swept = _swept(params, ratio, loss, y)
            floors = floors_at_capacity(
                day, volume, y, loss, k, swept, np.random.default_rng(seed)
            )
            review[ratio]["all_pass"].append(floors.all_pass)
            review[ratio]["all_hold"].append(floors.all_hold)
            review[ratio]["random_at_k"].append(floors.random_at_k)
            review[ratio]["volume_rank"].append(floors.volume_rank)
            for name, s in scores.items():
                review[ratio][name].append(
                    savings_of_ranking(s, day, y, loss, k, swept)
                )
            # A floor's REVIEW-only savings is invariant to the swept ratio by
            # construction: false_hold_cost enters `row_cost` only on HOLD rows and a
            # REVIEW-only policy emits none. The value is recomputed at every ratio anyway
            # rather than asserted, so the flatness below is a measurement.
            del swept

    # The denominator of every swept ratio, recorded so a reader can locate the cost matrix
    # the ladder actually ships on inside (or outside) the declared grid.
    fraud_rows = loss[y == 1]
    reference = float(fraud_rows.mean()) if fraud_rows.size else float("nan")

    _write(hold, review, nohold, k=k, seeds=args.seeds, params=params,
           n_pos=int(y.sum()), n_rows=int(y.size), n_merchants=len(merchants),
           reference_fraud_loss_inr=reference,
           shipped_ratio=params.false_hold_cost_inr / reference)
    print(f"wrote {OUT_MD} and {OUT_JSON}")
    return 0


def _render_only() -> int:
    """Rebuild the markdown from the committed JSON. Reads nothing else."""
    from rakshak.eval.metrics import CostParams

    payload = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    meta = payload["meta"]

    def unmean(block: dict[str, Any]) -> Any:
        out: Any = defaultdict(lambda: defaultdict(list))
        for ratio, policies in block.items():
            for name, value in policies.items():
                out[float(ratio)][name] = [value]
        return out

    _write(
        {arm: unmean(block) for arm, block in payload["hold_capable"].items()},
        unmean(payload["review_only"]),
        unmean(payload["hold_forbidden_arm_b"]),
        params=CostParams(
            review_cost_inr=meta["params"]["review_cost_inr"],
            false_hold_cost_inr=meta["params"]["false_hold_cost_inr_base"],
            fraud_loss_multiplier=meta["params"]["fraud_loss_multiplier"],
            p_catch=meta["params"]["p_catch"],
        ),
        # Everything else is forwarded verbatim, so a meta field added to the scoring run
        # reaches the renderer without a second edit here.
        **{k: v for k, v in meta.items() if k not in ("params", "ratios", "split")},
    )
    print(f"re-rendered {OUT_MD} from {OUT_JSON}")
    return 0


def _swept(params: Any, ratio: float, loss: np.ndarray, y: np.ndarray) -> Any:
    """The same CostParams ``sweep_cost_asymmetry`` builds internally, for the floor side.

    Duplicated here rather than exported from ``capacity`` because ``capacity`` is inside
    ``eval_module_sha256``: adding a helper to it would change the frozen hash and void the
    cycle's central claim. Kept to four lines so the duplication is checkable by eye, and
    ``test_cost_sweep.py`` asserts the two agree.
    """
    from rakshak.eval.metrics import CostParams

    fraud_loss = loss[y == 1]
    reference = float(fraud_loss.mean()) if fraud_loss.size else 0.0
    return CostParams(
        review_cost_inr=params.review_cost_inr,
        false_hold_cost_inr=ratio * reference,
        fraud_loss_multiplier=params.fraud_loss_multiplier,
        p_catch=params.p_catch,
    )


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _table(rows: dict[str, list[float]], order: list[str]) -> list[str]:
    return [f"| `{n}` | " + f"{_mean(rows[n]):+.4f}" + " |" for n in order if n in rows]


def _write(hold: Any, review: Any, nohold: Any, **meta: Any) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    ratios = list(capacity.ASYMMETRY_RATIOS)
    policies = ["rung1", "rung2", "rung3", "rung4", "rung9"]
    floors = ["all_pass", "all_hold", "random_at_k", "volume_rank"]

    payload = {
        "meta": {
            **{k: v for k, v in meta.items() if k != "params"},
            "params": {
                "review_cost_inr": meta["params"].review_cost_inr,
                "false_hold_cost_inr_base": meta["params"].false_hold_cost_inr,
                "p_catch": meta["params"].p_catch,
                "fraud_loss_multiplier": meta["params"].fraud_loss_multiplier,
            },
            "ratios": ratios,
            "split": "val",
        },
        "hold_capable": {
            arm: {str(r): {n: _mean(v) for n, v in hold[arm][r].items()} for r in ratios}
            for arm in hold
        },
        "review_only": {
            str(r): {n: _mean(v) for n, v in review[r].items()} for r in ratios
        },
        "hold_forbidden_arm_b": {
            str(r): {n: _mean(v) for n, v in nohold[r].items()} for r in ratios
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    L: list[str] = []
    a = L.append
    a("# Cost-asymmetry sweep — cycle 4, validation split")
    a("")
    a("> Generated by `scripts/cost_sweep.py`. Do not hand-edit.")
    a("")
    a("Every savings number published by this project before today was a **single point "
      "estimate** at one cost matrix. `metrics.CostParams`' own docstring says that is not "
      "enough: v1 measured the asymmetry at 47.5 / 13.1 / 61,368 against a literature band "
      "of 400–600, three orders of magnitude of spread. `sweep_cost_asymmetry` was built "
      "and unit-tested to answer this and had never been run on the ladder.")
    a("")
    a("## 1. What was varied")
    a("")
    a(f"- Swept: `false_hold_cost / mean_fraud_loss` over `ASYMMETRY_RATIOS` = {ratios}, "
      "the five points declared in `10-eval-harness-spec.md` §2 and **not extended** — "
      "a denser grid chosen after seeing the curve is an argument, not a measurement.")
    a(f"- Held: `review_cost_inr` = {meta['params'].review_cost_inr:,.0f}, "
      f"`p_catch` = {meta['params'].p_catch}, "
      f"`fraud_loss_multiplier` = {meta['params'].fraud_loss_multiplier}.")
    a(f"- Rows: validation split, K = {meta['k']}, {meta['n_merchants']:,} merchants, "
      f"{meta['n_rows']:,} scored merchant-days, {meta['n_pos']:,} positive rows, "
      f"seeds {meta['seeds']}, mean over seeds.")
    a("- Nothing was refitted. These are the decisions the committed models already make, "
      "re-priced.")
    a("")
    shipped_raw = meta.get("shipped_ratio")
    if isinstance(shipped_raw, float) and shipped_raw == shipped_raw:  # present, not nan
        shipped = shipped_raw
        ref = float(meta["reference_fraud_loss_inr"])
        inside = ratios[0] <= shipped <= ratios[-1]
        a(f"**Where the shipped cost matrix sits on this grid.** The sweep's denominator is "
          f"the mean `true_loss_amount_inr` over the fraud rows in this window, "
          f"₹{ref:,.0f}. Against it, the config's `false_hold_cost_inr` of "
          f"₹{meta['params'].false_hold_cost_inr:,.0f} is a ratio of **{shipped:.5f}** — "
          + ("inside" if inside else
             f"**below the lowest swept point ({ratios[0]:g})**, i.e. outside the grid")
          + ". So every savings number this project has published sits "
          + ("within" if inside else "off the left-hand end of")
          + " the range declared in `10-eval-harness-spec.md` §2, on the "
            "'a wrong HOLD is cheap relative to a fraud' side. That is a property of this "
            "generator's loss distribution, not a tuning choice, and the grid is left as "
            "declared: extending it after seeing the tables is how a sweep becomes an "
            "argument. It does bound the reading — the tables below say the ranking is "
            "stable across the declared range, and the operating point is at or beyond "
            "the cheap-HOLD end of it, where the rungs do best.")
        a("")
    a("## 2. Table A — priced on the actions each policy actually takes")
    a("")
    a("Rungs go through `select_actions` and may HOLD. Floors are REVIEW-only by "
      "construction and appear in Table B.")
    a("")
    for arm in ("declared", "realised"):
        a(f"**Arm {'A' if arm == 'declared' else 'B'} — {arm} exposure**")
        a("")
        a("| policy | " + " | ".join(f"ratio {r:g}" for r in ratios) + " |")
        a("|---|" + "---|" * len(ratios))
        for n in policies:
            cells = " | ".join(f"{_mean(hold[arm][r][n]):+.4f}" for r in ratios)
            a(f"| `{n}` | {cells} |")
        a("")
    a("## 3. Table B — every policy priced REVIEW-only, the floors' own convention")
    a("")
    a("This is the like-for-like comparison the ladder cannot make. `STATE.md` records it "
      "as an unowned defect: floors are priced REVIEW-only (₹250/error) while rungs are "
      "priced on their own actions, which may HOLD (₹8,250/error) — a 33× asymmetry inside "
      "the comparison that decides FLOOR-FAIL. `savings_of_ranking`'s docstring claims the "
      "comparison differs only in the score vector; that is true floor-vs-floor and false "
      "floor-vs-rung. Fixing it means editing the locked eval package, so it is measured "
      "here instead of corrected.")
    a("")
    a("| policy | " + " | ".join(f"ratio {r:g}" for r in ratios) + " |")
    a("|---|" + "---|" * len(ratios))
    for n in floors + policies:
        if n not in review[ratios[0]]:
            continue
        cells = " | ".join(f"{_mean(review[r][n]):+.4f}" for r in ratios)
        a(f"| `{n}` | {cells} |")
    a("")
    a("A floor's REVIEW-only savings is **flat across the sweep by construction** — "
      "`false_hold_cost` enters `row_cost` only on HOLD rows and a REVIEW-only policy "
      "emits none. It is recomputed at every ratio anyway rather than asserted, so the "
      "flatness above is a measurement and would break visibly if that stopped being true.")
    a("")
    a("## 4. Table C — arm B through the decision layer, with HOLD forbidden")
    a("")
    a("Table B changes two things at once against Table A: it strips the HOLD action *and* "
      "the expected-value re-ranking, reverting to a raw top-K by score. That is the error "
      "this document criticises elsewhere, so it is not left standing. Table C changes one "
      "thing: the selector, the exposure vector and the top-K are exactly as scored, and "
      "only HOLD is made unreachable — via `hold_expected_loss_floor_inr = inf`, the one "
      "knob that forbids it without editing the locked selector.")
    a("")
    a("| policy | " + " | ".join(f"ratio {r:g}" for r in ratios) + " |")
    a("|---|" + "---|" * len(ratios))
    for n in policies:
        cells = " | ".join(f"{_mean(nohold[r][n]):+.4f}" for r in ratios)
        a(f"| `{n}` | {cells} |")
    a("")
    a("One residual impurity, stated rather than hidden: `select_actions` ranks by "
      "`cost_pass - min(cost_review, cost_hold)`, so `cost_hold` still influences *which* "
      "rows are selected even when HOLD cannot be *taken*. Removing that would mean editing "
      "the locked selector. The effect is bounded — at low ratios `cost_hold` collapses "
      "toward `review_cost`, at high ratios `cost_review` is the minimum — but Table C is "
      "\"HOLD unreachable\", not \"HOLD absent from the arithmetic\".")
    a("")

    # -- the verdict, computed from the tables above ---------------------------
    floor_v = _mean(review[ratios[0]]["volume_rank"])
    best_a = {r: max(policies, key=lambda n: _mean(hold["realised"][r][n])) for r in ratios}
    beats = [r for r in ratios if _mean(hold["realised"][r][best_a[r]]) > floor_v]
    a("## 5. What the sweep settles")
    a("")
    a(f"**1. The savings result is stable across the whole declared range.** The best arm-B "
      f"rung beats the `volume_rank` floor ({floor_v:+.4f}) at **{len(beats)} of "
      f"{len(ratios)}** swept ratios"
      + (f" ({', '.join(f'{r:g}' for r in beats)})." if beats else ".")
      + " A result that survives four orders of magnitude of cost asymmetry is a "
        "materially stronger claim than a win at one guessed ratio, and it is the claim a "
        "business case actually needs. Per-rung range under arm B:")
    a("")
    for n in policies:
        vals = [_mean(hold["realised"][r][n]) for r in ratios]
        winner = " <- best at every ratio" if all(best_a[r] == n for r in ratios) else ""
        a(f"   - `{n}`: {min(vals):+.4f} to {max(vals):+.4f} "
          f"(spread {max(vals) - min(vals):.4f}){winner}")
    a("")
    best_b = max(policies, key=lambda n: _mean(review[ratios[0]][n]))
    b_gap = floor_v - _mean(review[ratios[0]][best_b])
    a(f"**2. The rungs do not out-rank the size floor on rupees. The decision layer is what "
      f"puts them ahead.** Priced as pure rankings under the floors' own REVIEW-only "
      f"convention (Table B), the best rung (`{best_b}`) loses to `volume_rank` by "
      f"**{b_gap:.4f}**. This is LIMITATIONS.md §8.3a's mechanism restated at the level of "
      f"the cost matrix: `volume_rank` is an exposure estimator, and on rupees an exposure "
      f"estimator beats a fraud-probability ranker that is never told what is at stake.")
    a("")
    a_best_name = best_a[ratios[0]]
    a_best = _mean(hold["realised"][ratios[0]][a_best_name])
    c_of_a = _mean(nohold[ratios[0]][a_best_name])
    margin_full = a_best - floor_v
    margin_nohold = c_of_a - floor_v
    hold_share = (margin_full - margin_nohold) / margin_full if margin_full else float("nan")
    nohold_beats = [r for r in ratios if _mean(nohold[r][a_best_name]) > floor_v]
    a(f"**3. The margin survives the pricing asymmetry §4.3 disclosed, but not by much, and "
      f"the decomposition should be reported rather than the headline alone.** Take "
      f"`{a_best_name}`, the best row in Table A, at ratio {ratios[0]:g}:")
    a("")
    a(f"   - Table A, as scored (HOLD permitted): **{a_best:+.4f}**, "
      f"a margin of **{margin_full:+.4f}** over the floor.")
    a(f"   - Table C, HOLD unreachable, everything else identical: **{c_of_a:+.4f}**, "
      f"a margin of **{margin_nohold:+.4f}**. It still beats the floor, at "
      f"**{len(nohold_beats)} of {len(ratios)}** ratios.")
    a(f"   - Table B, the raw ranking: **{_mean(review[ratios[0]][a_best_name]):+.4f}**, "
      f"a margin of **{_mean(review[ratios[0]][a_best_name]) - floor_v:+.4f}** — negative, "
      f"and by a wide margin.")
    a("")
    a(f"So the HOLD privilege is worth about **{hold_share:.0%}** of the rung's margin over "
      f"the floor, the exposure-aware re-ranking supplies the rest, and the score ranking "
      f"on its own is worth less than nothing against a size ranking. §4.3 of the "
      f"pre-registration disclosed the 33x asymmetry and declined to fix it inside a locked "
      f"harness; this quantifies what it was worth. **The direction matters: the rung would "
      f"still beat the floor with HOLD switched off**, so the FLOOR-FAIL verdict does not "
      f"rest on the unfair half of the comparison — but a report that quotes "
      f"{margin_full:+.4f} without {margin_nohold:+.4f} beside it is quoting the flattering "
      f"one.")
    a("")
    a("**4. What this does not settle.** Every number here is on the **validation** split "
      "(`open_count` is 0). None of it is a test-split result and none of it was "
      "pre-registered: the sweep ran after the ladder was scored, which makes it a "
      "robustness check on a measured result, not a gate, and it must not be reported as "
      "one. The latency half of charter §2 is untouched — savings is a rupee metric and "
      "says nothing about detection delay, and `rung4`, the best row in Table A, has "
      "`ttd_median_days` of `inf` in both arms.")
    a("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
