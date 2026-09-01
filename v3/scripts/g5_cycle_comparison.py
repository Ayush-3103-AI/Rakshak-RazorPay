"""G5 under the cycle-3 config and the cycle-4 config, measured identically.

`docs/gates/GATES-CYCLE4.md` records an observation it explicitly declines to call a
finding: on cycle-4 data the RAW detector is RED at +7.07pp worst-window excess while the
cohort residual is RED at +2.70pp, where `LIMITATIONS.md` §6 reports the raw detector GREEN
for cycle 3 and concludes *"the demo premise does not hold."* That comparison is between two
different runs of two different configs and is not controlled: G5 runs at `prevalence=0`, so
the onset-window change cannot act on it directly, but it does move the RNG stream.

This script runs the **same measurement** over both configs at the same gate seed and the
same gate population, so the two numbers differ by the config alone.

    uv run python scripts/g5_cycle_comparison.py --cycle3-config <path-to-cycle-3.yaml>

Get the cycle-3 config out of the tag rather than reconstructing it:

    git show cycle3-ladder-immutable:v3/configs/scenario_v2.yaml > /tmp/cycle3.yaml

`configs/scenario_v2.yaml` is **not touched**. `tests/gates/gates_report.py::CONFIG_PATH` is
a hardcoded constant, so the gate itself can only ever measure the live config; this builds
its scenarios directly instead, reusing the gate's own `trailing_z`, `cohort_residual`,
`calibrate` and `alert_rate` so the statistic cannot drift from the one the gate reports.

**A caveat this script cannot remove.** The cycle-3 config has no
`labels.label_resolution_horizon_day`, which `LabelsConfig` now requires. It is injected at
its backward-compatible value (`n_days - 1`, the pre-cycle-4 censoring rule) so the load
succeeds. At `prevalence=0` no label is drawn at all, so the injection cannot affect the
measurement — but it does mean the file being loaded is not byte-identical to the one cycle
3 shipped, and that is stated rather than hidden.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "gates"))

from gates_report import GATE_MERCHANTS, GATE_SEED, daily_counts  # noqa: E402
from test_g5_confounder_null import (  # noqa: E402
    BASELINE_DAYS,
    EXCESS_ALLOWED,
    alert_rate,
    calibrate,
    cohort_residual,
    trailing_z,
)

from rakshak.generator.config import ScenarioConfig, load_scenario  # noqa: E402
from rakshak.generator.confounders import build_layer  # noqa: E402
from rakshak.generator.engine import generate  # noqa: E402


def load_null_scenario(path: Path) -> ScenarioConfig:
    """The gate's own population overrides, applied to an arbitrary manifest."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels = raw.setdefault("labels", {})
    if "label_resolution_horizon_day" not in labels:
        # Backward-compatible value: the pre-cycle-4 rule. Inert at prevalence=0.
        labels["label_resolution_horizon_day"] = int(raw["population"]["n_days"]) - 1
    tmp = path.with_suffix(".resolved.yaml")
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    try:
        cfg = load_scenario(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return dataclasses.replace(
        cfg,
        population=dataclasses.replace(
            cfg.population, n_merchants=GATE_MERCHANTS, prevalence=0.0
        ),
    )


def measure(cfg: ScenarioConfig, label: str) -> dict[str, float]:
    n_days = cfg.population.n_days
    data = generate(cfg, np.random.default_rng(GATE_SEED + 1))
    layer = build_layer(cfg, np.zeros(4, dtype=np.int64), np.full(4, 6.0), np.full(4, 0.5))
    nominal = cfg.capacity.analyst_reviews_per_day / cfg.capacity.per_n_merchants
    counts = daily_counts(data, n_merchants=GATE_MERCHANTS, n_days=n_days)

    busy = np.zeros(n_days, dtype=bool)
    for w in layer.windows:
        busy[w.start_day : w.end_day] = True
    quiet = np.flatnonzero(~busy & (np.arange(n_days) > BASELINE_DAYS))

    out: dict[str, float] = {}
    raw = trailing_z(counts, BASELINE_DAYS)
    for name, z in (("raw", raw), ("cohort-residual", cohort_residual(raw))):
        threshold = calibrate(z, quiet, nominal)
        worst = 0.0
        for w in layer.windows:
            days = np.arange(w.start_day, w.end_day)
            days = days[days > BASELINE_DAYS]
            if days.size:
                worst = max(worst, alert_rate(z, threshold, days) - nominal)
        out[name] = worst * 100.0
    print(f"  {label:<28} n_days={n_days}  quiet days={quiet.size}  nominal={nominal:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle3-config", type=Path, required=True)
    ap.add_argument("--cycle4-config", type=Path,
                    default=Path("configs/scenario_v2.yaml"))
    args = ap.parse_args()
    if not args.cycle3_config.exists():
        print(f"no config at {args.cycle3_config}; see this script's docstring")
        return 1

    print(f"G5 worst-window excess, same seed ({GATE_SEED + 1}), same population "
          f"({GATE_MERCHANTS:,}), prevalence=0\n")
    c3 = measure(load_null_scenario(args.cycle3_config), "cycle-3 config")
    c4 = measure(load_null_scenario(args.cycle4_config), "cycle-4 config")

    print(f"\n  {'detector':<20}{'cycle 3':>12}{'cycle 4':>12}{'verdict':>28}")
    for det in ("raw", "cohort-residual"):
        v3, v4 = c3[det], c4[det]
        verdict = (f"{'GREEN' if v3 <= EXCESS_ALLOWED * 100 else 'RED'}"
                   f" -> {'GREEN' if v4 <= EXCESS_ALLOWED * 100 else 'RED'}")
        print(f"  {det:<20}{v3:>+11.2f}pp{v4:>+11.2f}pp{verdict:>28}")

    print(f"\n  allowance is +{EXCESS_ALLOWED * 100:.0f}pp above nominal.")
    gain3 = c3["raw"] - c3["cohort-residual"]
    gain4 = c4["raw"] - c4["cohort-residual"]
    print(f"  residual's advantage over raw: cycle 3 {gain3:+.2f}pp, cycle 4 {gain4:+.2f}pp")
    print("\n  Read this against LIMITATIONS.md §6. If the residual's advantage is large in")
    print("  BOTH columns, §6's cycle-3 conclusion was measuring something else. If it is")
    print("  large only in cycle 4, the population size is doing the work and that is the")
    print("  finding. If neither, §6 stands and the GATES-CYCLE4.md observation was noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
