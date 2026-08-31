"""The v2 generator: personas x typologies x confounders, NB/Hawkes arrivals, delayed labels.

**This is an evaluation artifact, not a fraud toolkit.** It emits risk typologies so that
detection can be measured against them, it lives only in this package, and it is documented
as such in the README. The track is strictly defense-only.

May read and write GroundTruth (it is the thing that produces it). Must never be imported
from rakshak.features or rakshak.models -- Prime Directive 3.
"""
