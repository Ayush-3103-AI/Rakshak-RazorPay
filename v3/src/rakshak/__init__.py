"""Rakshak v2 — post-onboarding merchant risk sentinel.

v2 is a separate cycle from v1 with its own generator, its own eval harness, and its own
lock. v1's numbers are immutable and live under the `v1-frozen` git tag; nothing here may
edit, re-run or "correct" them.
"""

__version__ = "2.0.0"

# Bumped whenever a persisted table's shape changes. Every persisted row carries it, so a
# parquet file written under an older schema fails loudly instead of being read wrong.
SCHEMA_VERSION = 1
