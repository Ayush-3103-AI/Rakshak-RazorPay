"""Shim. Rung 7's runner moved into the package as ``rakshak.explain.hsmm_onset``.

It had to move: an explainer that only a loose script can reach is an explainer nobody
runs, and ``rakshak.cli explain`` cannot import from ``scripts/``. The move also put the
registration (``explain.registry.register``) next to the thing being registered, and moved
the artifact out of ``data/v2/eval/`` — that directory is the glob
``artifacts/build.py::read_result_rows`` turns into ``ladder.json``, and Rung 7 has no
ladder row by design.

Kept as a shim rather than deleted so an existing `uv run python scripts/rung7_score.py`
still works.
"""

from __future__ import annotations

from rakshak.explain.hsmm_onset import main

if __name__ == "__main__":
    main()
