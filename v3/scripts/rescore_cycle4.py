"""Rescore the whole cycle-4 ladder: every floor, every rung, both exposure arms.

Pre-registration §4.2 requires the A/B to run over the entire ladder on identical scores,
and §6 requires all five seeds on every row. That is a large matrix and running it by hand
is how a row ends up missing or scored under the wrong arm. This driver is the matrix,
written down.

It does not compute anything itself. It shells out to the same ``rakshak.cli`` entry points
a human would type, so there is no second code path that could drift from the documented
one, and every command it runs is printed.

    uv run python scripts/rescore_cycle4.py --plan          # print the matrix, run nothing
    uv run python scripts/rescore_cycle4.py --jobs 4        # run it

**The test split is not reachable from here.** ``--split`` is hard-wired to ``val``. Opening
the test split is a one-way door governed by PRE-REGISTRATION-CYCLE4 §5 and it is not
something a batch driver should be able to do by accident.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)

#: Floors do not consume ``exposure_inr`` at all — ``volume_rank`` ranks on observed volume
#: and all three act REVIEW-only through ``floor_actions``. They are therefore the same
#: constant in both arms, which is exactly what makes them a usable reference line. They are
#: rescored on cycle-4 data at the new K, which is the substantive part.
FLOORS = ("all_pass", "random_at_k", "volume_rank")

#: Rungs that go through ``select_actions`` and are therefore sensitive to the arm.
#: Rung 1 needs no training. 5/6/7 are dispatched differently and are appended by
#: ``--with-upper`` once the base ladder is down.
TRAINED = (2, 3, 4)


@dataclass(frozen=True)
class Job:
    kind: str
    argv: tuple[str, ...]
    label: str

    def run(self) -> tuple[str, int, float, str]:
        started = time.perf_counter()
        proc = subprocess.run(  # noqa: S603
            self.argv, cwd=ROOT, capture_output=True, text=True
        )
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return self.label, proc.returncode, time.perf_counter() - started, (
            tail[-1][:200] if tail else ""
        )


def _py() -> tuple[str, ...]:
    return (sys.executable, "-m", "rakshak.cli")


def train_jobs() -> list[Job]:
    return [
        Job("train", (*_py(), "train", "--rung", str(r), "--seed", str(s)),
            f"train rung{r} seed{s}")
        for r in TRAINED
        for s in SEEDS
    ]


def eval_jobs(*, with_upper: bool) -> list[Job]:
    jobs: list[Job] = []
    for s in SEEDS:
        for f in FLOORS:
            jobs.append(Job(
                "eval",
                (*_py(), "eval", "--rung", "0", "--floor", f, "--seed", str(s),
                 "--split", "val"),
                f"eval {f} seed{s}",
            ))
        for r in (1, *TRAINED):
            for arm in ("declared", "realised"):
                jobs.append(Job(
                    "eval",
                    (*_py(), "eval", "--rung", str(r), "--seed", str(s),
                     "--split", "val", "--exposure", arm),
                    f"eval rung{r} seed{s} [{arm}]",
                ))
        if with_upper:
            for r in (5, 6):
                jobs.append(Job(
                    "eval",
                    (*_py(), "eval", "--rung", str(r), "--seed", str(s), "--split", "val"),
                    f"eval rung{r} seed{s}",
                ))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=3,
                    help="Concurrent processes. Training is memory-hungry; 3 is safe on "
                         "16 GB. Evaluation is lighter.")
    ap.add_argument("--plan", action="store_true", help="Print the matrix and exit.")
    ap.add_argument("--skip-train", action="store_true",
                    help="Models already trained on cycle-4 data.")
    ap.add_argument("--with-upper", action="store_true",
                    help="Also rescore Rungs 5 and 6.")
    args = ap.parse_args()

    phases: list[tuple[str, list[Job]]] = []
    if not args.skip_train:
        phases.append(("TRAIN", train_jobs()))
    phases.append(("EVAL", eval_jobs(with_upper=args.with_upper)))

    if args.plan:
        for name, jobs in phases:
            print(f"\n--- {name}: {len(jobs)} jobs ---")
            for j in jobs:
                print("   ", " ".join(j.argv[2:]))
        print(f"\ntotal {sum(len(j) for _, j in phases)} jobs")
        return 0

    failures: list[tuple[str, str]] = []
    for name, jobs in phases:
        # Training must complete before evaluation: an eval that races its own model reads
        # a half-written artefact, and the failure looks like a bad number rather than a
        # bad file. Within a phase the jobs are independent.
        print(f"\n=== {name} — {len(jobs)} jobs, {args.jobs} at a time ===", flush=True)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for label, code, secs, tail in pool.map(lambda j: j.run(), jobs):
                mark = "ok  " if code == 0 else "FAIL"
                print(f"  {mark} {label:38} {secs:7.1f}s  {tail if code else ''}",
                      flush=True)
                if code:
                    failures.append((label, tail))
        print(f"=== {name} done in {(time.perf_counter() - started) / 60:.1f} min ===")

    if failures:
        print(f"\n{len(failures)} FAILED:")
        for label, tail in failures:
            print(f"  {label}: {tail}")
        return 1
    print("\nall jobs green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
