"""Rakshak generator layer — synthetic merchant streams for EVALUATION ONLY.

See `rakshak.generator.generate` for the full scope-and-safety statement (FR-006): this is a
defensive measurement artifact, not a fraud toolkit, and it produces nothing usable against a
live payment system.
"""

from rakshak.generator.generate import (
    MIN_ONSET_DAY,
    MIN_POST_ONSET_DAYS,
    NO_TYPOLOGY,
    SEGMENTS,
    STATE_PATH_COLUMNS,
    STATES,
    TRANSACTION_COLUMNS,
    TYPOLOGIES,
    GeneratorConfig,
    generate,
    onset_window,
    write_outputs,
)

__all__ = [
    "NO_TYPOLOGY",
    "SEGMENTS",
    "STATES",
    "STATE_PATH_COLUMNS",
    "TRANSACTION_COLUMNS",
    "TYPOLOGIES",
    "MIN_ONSET_DAY",
    "MIN_POST_ONSET_DAYS",
    "GeneratorConfig",
    "generate",
    "onset_window",
    "write_outputs",
]
