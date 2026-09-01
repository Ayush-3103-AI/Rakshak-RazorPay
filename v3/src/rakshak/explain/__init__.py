"""Explainers: things that say *why*, and are structurally barred from saying *whether*.

CLAUDE.md's audience section names the third question a Head of Risk Ops asks — "can I
explain the decision to the merchant when they call and shout?" — as the one nobody else
in the submission pool will answer. This package answers it.

The registration surface lives in :mod:`rakshak.explain.registry`. The explainers
themselves do not exist yet, and that is the intended state at T-0118: the pre-registration
(docs/PRE-REGISTRATION-CYCLE3-2026-08-31.md §4) commits to landing the eval-side surface
with **no rung attached**, so the cycle-3 lock can hash it before Rungs 5-8 are written.
"""
