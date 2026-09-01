# Lock verification record — taken before T-0118 lands

**Date:** 2026-09-01 · **Author:** lead · **Tree verified:** `5167abb`

## Why this file exists

`EVAL_MODULES` — the tuple whose contents `eval_module_sha256` covers — is:

```
src/rakshak/eval/splits.py
src/rakshak/eval/metrics.py
src/rakshak/eval/oracle.py
src/rakshak/eval/capacity.py
src/rakshak/eval/lock.py
```

T-0118 edits three of those five: `metrics.py` (the three cycle-3 metrics),
`capacity.py` (the decision-policy seam), and `lock.py` (the three metric names enter
`write_lock`'s declared list).

The cycle-3 pre-registration §4 anticipates this — it is the whole reason the cycle-3 lock
must be written *after* T-0118 rather than before. What §4 does **not** say is the
consequence for the two locks that are already sealed: **their `eval_module_sha256` stops
matching the working tree the moment T-0118 lands, and it never matches again.** `lock.py`
does not return to its old bytes.

Without this record, the claim "Rungs 0-4 were judged on the harness as sealed" would rest
on nothing checkable after today, and the drift would surface later as an anomaly rather
than as an intended, dated consequence.

## The verification, taken while it was still takeable

At `5167abb`, with T-0118's work present in the working tree but **not** committed, the
committed `HEAD` tree still held the original five modules. Reconstructing them with
`git show HEAD:v3/src/rakshak/eval/<m>.py` and hashing with `hash_paths`:

| Lock | `open_count` | sealed `eval_module_sha256` | committed tree at `5167abb` | result |
|---|---|---|---|---|
| `EVAL-LOCK.json` (cycle 1) | 0 | `f15be39ef68343b26eb6…` | `f15be39ef68343b26eb6…` | **VERIFIED** |
| `EVAL-LOCK-CYCLE2.json` (cycle 2) | 0 | `f15be39ef68343b26eb6…` | `f15be39ef68343b26eb6…` | **VERIFIED** |

Both sealed locks hash the same eval modules — cycle 2 re-sealed the harness unchanged.
`open_count` is 0 on both. `RAKSHAK_UNLOCK` was not set. No lock file was written or
edited; both are byte-identical to `HEAD`.

## How to verify cycles 1 and 2 from here on

Not against the working tree — that check is expected to fail from T-0118 onward, and a
failure there is **not** evidence of tampering. Verify against the tree at `5167abb`:

```sh
for m in splits metrics oracle capacity lock; do
  git show 5167abb:v3/src/rakshak/eval/$m.py > "$T/src/rakshak/eval/$m.py"
done
# then hash_paths([T/m for m in EVAL_MODULES], T) == the sealed value above
```

## What this record does NOT claim

- It does not pre-approve any change to `metrics.py`, `capacity.py` or `lock.py`. It fixes
  a reference point; it does not bless what comes after it.
- It states no post-T-0118 hash. T-0118 was still in flight when this was written, so any
  live value measured today is transient and is deliberately omitted rather than recorded
  as if final. The cycle-3 lock records the settled value when it is sealed.
- It does not touch the sealed locks, and it is not itself a lock.
