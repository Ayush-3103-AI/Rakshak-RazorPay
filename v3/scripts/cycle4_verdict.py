"""Render the cycle-4 verdict against the gates PRE-REGISTRATION-CYCLE4 declared.

Every threshold here is read from the pre-registration, not chosen here. The point of this
script is that the verdict is computed rather than eyeballed: a gate assessed by reading a
table is a gate that moves when the table is disappointing.

It answers, in order:

1. **Did the geometry fix work?**  ``detection_rate_d30`` non-zero for at least one policy
   on cycle-4 validation. This is §5 condition 3 and the falsifier in §7 row 1.
2. **Did the exposure finding survive contact with new data?**  Arm B raises savings for
   the scoring rungs. §7 row 3 — if it does not, `LIMITATIONS.md` §8.3a is wrong and gets
   reported as falsified.
3. **Is the floor-fail closed?**  Best rung under arm B reaches savings ≥ 0.7017 at ≥ 4/5
   seeds. §4.2.
4. **Is the win degenerate?**  `alert_jaccard_wow < 0.95` and `alerts_per_day ≥ 0.9·K`. §4.2.
5. **Were the two failures one failure?**  §7 row 4: does arm A close the floor-fail on its
   own, with no exposure correction? If it does, the stationary-window hypothesis explains
   the cycle-3 result and the exposure finding, while still true as arithmetic, was not what
   decided it.
6. **Does the test split open?**  All four §5 conditions, reported individually.

    uv run python scripts/cycle4_verdict.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "v2" / "eval"
SEEDS = (42, 43, 44, 45, 46)

# ── the pre-registered numbers. Changing one of these is amending a sealed document. ──
SAVINGS_GATE = 0.7017  # §4.2: the cycle-3 volume_rank floor 0.6017 + 0.10 absolute
MIN_SEEDS = 4  # §4.2: holding at >= 4 of 5 seeds
MAX_JACCARD = 0.95  # §4.2 anti-degeneracy
MIN_ALERT_FRACTION = 0.9  # §4.2 anti-degeneracy
CYCLE3_VOLUME_RANK_SAVINGS = 0.6017  # the immutable reference line


def load() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """``(label, arm) -> [row per seed]``. Arm is read off the row, never inferred."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(EVAL_DIR.glob("*_val_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        label = str(row.get("label", path.stem))
        arm = str(row.get("exposure_arm", "declared"))
        base = label.removesuffix("_realised_exposure")
        out[(base, arm)].append(row)
    return out


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]
    return statistics.fmean(vals) if vals else None


def _fmt(v: float | None, nd: int = 4) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    if not EVAL_DIR.exists():
        print(f"no results at {EVAL_DIR}; run scripts/rescore_cycle4.py first")
        return 1
    data = load()
    if not data:
        print(f"no *_val_seed*.json under {EVAL_DIR}")
        return 1

    print("=" * 96)
    print("CYCLE-4 LADDER — validation, mean over seeds, both exposure arms")
    print("=" * 96)
    cols = ("pr_auc", "ece", "savings", "precision_at_k", "recall_at_k",
            "alerts_per_day", "alert_jaccard_wow", "detection_rate_d30", "ttd_median_days")
    print(f"{'policy':<22}{'arm':<10}{'n':>3}" + "".join(f"{c[:13]:>15}" for c in cols))
    for (label, arm) in sorted(data):
        rows = data[(label, arm)]
        print(f"{label:<22}{arm:<10}{len(rows):>3}"
              + "".join(f"{_fmt(_mean(rows, c)):>15}" for c in cols))

    def best(arm: str) -> tuple[str, list[dict[str, Any]]] | None:
        scoring = {k: v for k, v in data.items()
                   if k[1] == arm and not k[0].startswith(("all_", "random_", "volume_"))}
        if not scoring:
            return None
        k = max(scoring, key=lambda kk: _mean(scoring[kk], "savings") or -9e9)
        return k[0], scoring[k]

    print("\n" + "=" * 96)
    print("THE PRE-REGISTERED QUESTIONS")
    print("=" * 96)

    # 1 — the geometry fix
    d30 = [(k[0], k[1], _mean(v, "detection_rate_d30")) for k, v in data.items()]
    fired = [(lbl, arm, m) for lbl, arm, m in d30 if m is not None and m > 0.0]
    print("\n1. GEOMETRY — did detection_rate_d30 become measurable?")
    print(f"   {'YES' if fired else 'NO'}: {len(fired)} of {len(d30)} policies score "
          f"non-zero d30 (cycle 3: 0 of 7).")
    for lbl, arm, m in sorted(fired, key=lambda t: -(t[2] or 0))[:5]:
        print(f"      {lbl} [{arm}]  d30={m:.4f}")
    if not fired:
        print("      §7 row 1 FALSIFIED: the regeneration did not make TTD measurable.")

    # 2 — the exposure mechanism
    print("\n2. EXPOSURE — does arm B raise savings? (§7 row 3; falsifies §8.3a if not)")
    moved: list[tuple[str, float, float]] = []
    for (label, arm) in sorted(data):
        if arm != "declared" or (label, "realised") not in data:
            continue
        a = _mean(data[(label, "declared")], "savings")
        b = _mean(data[(label, "realised")], "savings")
        if a is not None and b is not None:
            moved.append((label, a, b))
            print(f"      {label:<18} declared={a:+.4f}  realised={b:+.4f}  "
                  f"delta={b - a:+.4f}  {'UP' if b > a else 'DOWN'}")
    if moved:
        up = sum(1 for _, a, b in moved if b > a)
        print(f"   {up} of {len(moved)} rungs improve under realised exposure.")
        if up == 0:
            print("      §8.3a FALSIFIED. Report it as falsified; do not explain it away.")

    # 3 & 4 — the floor-fail gate and anti-degeneracy
    print(f"\n3. FLOOR-FAIL — best rung under arm B vs the gate {SAVINGS_GATE:.4f}")
    bb = best("realised")
    gate3 = gate4 = False
    if bb:
        label, rows = bb
        per_seed = [r["savings"] for r in rows if r.get("savings") is not None]
        n_pass = sum(1 for s in per_seed if s >= SAVINGS_GATE)
        gate3 = n_pass >= MIN_SEEDS
        print(f"      {label}: mean={_fmt(_mean(rows, 'savings'))}, "
              f"per-seed={[round(s, 4) for s in per_seed]}")
        print(f"      {n_pass}/{len(per_seed)} seeds >= {SAVINGS_GATE} "
              f"(need {MIN_SEEDS}) -> {'PASS' if gate3 else 'FAIL'}")
        jac, alerts = _mean(rows, "alert_jaccard_wow"), _mean(rows, "alerts_per_day")
        k = _mean(rows, "capacity_k")
        ok_j = jac is None or jac < MAX_JACCARD
        ok_a = alerts is None or k is None or alerts >= MIN_ALERT_FRACTION * k
        gate4 = ok_j and ok_a
        print("\n4. ANTI-DEGENERACY — is the win a re-derived watchlist?")
        print(f"      alert_jaccard_wow={_fmt(jac)} < {MAX_JACCARD} -> "
              f"{'PASS' if ok_j else 'FAIL'}")
        print(f"      alerts_per_day={_fmt(alerts, 1)} >= {MIN_ALERT_FRACTION}*K"
              f"={_fmt(None if k is None else MIN_ALERT_FRACTION * k, 1)} -> "
              f"{'PASS' if ok_a else 'FAIL'}")

    # 5 — were the two failures one failure?
    print("\n5. ONE FAILURE OR TWO? — does arm A close the floor-fail on its own? (§7 row 4)")
    ba, vr = best("declared"), data.get(("volume_rank", "declared"))
    if ba and vr:
        a_best, vr_s = _mean(ba[1], "savings"), _mean(vr, "savings")
        print(f"      best rung arm A = {_fmt(a_best)} ({ba[0]});  "
              f"volume_rank cycle 4 = {_fmt(vr_s)};  cycle 3 = {CYCLE3_VOLUME_RANK_SAVINGS}")
        if a_best is not None and vr_s is not None:
            print("      -> " + (
                "arm A ALREADY beats the floor. The stationary-window hypothesis is "
                "supported: volume_rank was winning because the window held no onsets. "
                "Report the exposure finding as true arithmetic that was NOT what decided "
                "cycle 3."
                if a_best > vr_s else
                "arm A still loses to the floor, so in-window onsets alone do not close "
                "it. The two failures are not the same failure."))

    # 6 — the test split
    print("\n6. TEST SPLIT — the four §5 conditions")
    for i, (desc, ok) in enumerate((
        (f"best arm-B rung savings >= {SAVINGS_GATE} at >= {MIN_SEEDS}/5 seeds", gate3),
        ("both anti-degeneracy conditions hold", gate4),
        ("detection_rate_d30 non-zero for at least one policy", bool(fired)),
        ("lock verifies and eval_module_sha256 unchanged", None),
    ), start=1):
        mark = ("UNCHECKED (verify_lock)" if ok is None
                else ("PASS" if ok else "FAIL"))
        print(f"      {i}. {desc:<62} {mark}")
    print("\n   The split opens ONLY if all four pass. Otherwise it stays shut and the")
    print("   report says which condition failed. Conditionality is disclosed either way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
