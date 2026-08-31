"""The eval harness: splits, metrics, oracle, capacity, and the lock.

Frozen before any v2 model is written (Prime Directive 1). EVAL-LOCK.json is written once by
T-133 and the test split is opened exactly once, by T-151.

This is the only package permitted to read Label and GroundTruth, and
splits.available_labels(as_of) is the only path to the label table.
"""
