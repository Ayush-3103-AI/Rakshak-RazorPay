"""Rakshak eval layer — the frozen evaluation (06-requirements.md §3).

`splits` holds the leakage guard, `metrics` the frozen metric set, `oracle` the
perfect-foresight ceilings, `harness` the `make eval` entry point.
"""

from rakshak.eval.splits import BAD_STATES, Split, assert_no_leakage, load_split

__all__ = ["BAD_STATES", "Split", "assert_no_leakage", "load_split"]
