"""Shim. Rung 7's runner lives in the package as ``rakshak.score_rung7``.

It had to leave ``scripts/``: an explainer that only a loose script can reach is an
explainer nobody runs, and ``rakshak.cli explain`` cannot import from here. The move also
put the registration (``explain.registry.register``) next to the thing being registered,
and moved the artifact out of ``data/v2/eval/`` — that directory is the glob
``artifacts/build.py::read_result_rows`` turns into ``ladder.json``, and Rung 7 has no
ladder row by design.

It then had to leave ``explain/`` as well: the runner fits the HSMM, so it imports
``rakshak.models``, and ``test_explain_registry.py`` refuses that from anywhere under
``explain/``. The explainer itself stayed behind the wall in
``rakshak.explain.hsmm_onset``; only the runner moved. See ``score_rung7``'s docstring.

Kept as a shim rather than deleted so an existing `uv run python scripts/rung7_score.py`
still works.
"""

from __future__ import annotations

from rakshak.score_rung7 import main

if __name__ == "__main__":
    main()
