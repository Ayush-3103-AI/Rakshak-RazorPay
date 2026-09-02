# Rakshak G3 — the tree

> **The story, the results and the live panel are in the [repo README](../README.md).**
> This file is operational: what this tree is, how to run it, and the rules that govern it.
> It deliberately does not repeat a results table — one copy of a number, in the file that
> generates it, is what stops this repo's documents from drifting apart.

`ver-2/v3` is **G3**, the third generation. `CLAUDE.md` and
`project-context/00-charter-v2.md` call it **"v2"**, and call the tree at the repository root
**"v1"**; the separate [G1 repository](https://github.com/Ayush-3103-AI/razorpay-project) also
calls itself "v1". The public G1/G2/G3 labels exist to break that collision and are defined in
[`configs/journey.yaml`](configs/journey.yaml). No charter, lock or results file was edited to
introduce them.

## Prime directives

The full set is in [`CLAUDE.md`](CLAUDE.md). The two that constrain every session:

1. **The eval harness is frozen before any model is written.** The test split opens exactly
   once, at the end, and only if a pre-registered gate says so. It has never opened —
   `open_count` is 0 on all four locks. Debug on validation.
2. **Prior generations' results are immutable.** No G1 or G2 number is edited, re-run or
   "corrected". They are carried forward as cited literals.

## Commands

```bash
uv sync

make all      # lint → parity → gen → gates → perf → test. Must pass from a clean clone.
make gen      # regenerate the dataset from configs/scenario_v2.yaml
make gates    # G1–G5 generator parity gates. Must be green before any model trains.
make eval RUNG=n
make report   # regenerate docs/results_v2.md from the frozen eval
make artifacts    # emit artifacts/*.json — the dashboard's only data source
make lint     # ruff check + mypy --strict src/
make test     # pytest, all suites
```

**`make eval` refuses the locked test split unless `RAKSHAK_UNLOCK=1` is set.** It is not set
anywhere in this repo, and setting it is a deliberate, once-only act governed by the
pre-registration.

**`make all` from a clean `git clone` is a stop-work condition.** G1's single biggest
disqualification risk was `make eval` not reproducing on a fresh checkout, so
[`.github/workflows/v3-ci.yml`](../.github/workflows/v3-ci.yml) clones into a scratch path and
builds from nothing on every push. If it goes red, the sprint stops until it is green — do not
weaken `make all` to make it pass.

## Dashboard

A static React build over the committed artifacts. **No backend, ever** — the panel's only data
source is `artifacts/*.json`, and a missing or malformed artifact renders a named error rather
than a blank chart standing in for a number nobody measured.

```bash
cd dashboard
npm ci
npm run dev      # http://localhost:5173
npm test         # loader contract + a whole-app render against the real artifacts
npm run build    # -> dist/, deployed to GitHub Pages by ../.github/workflows/pages.yml
```

`predev`/`prebuild` copy `artifacts/` into `public/`, so dev and build serve the same bytes a
reader can open in the repo.

## Where things are

| | |
|---|---|
| [`LIMITATIONS.md`](LIMITATIONS.md) | Every failure, with the number. Prime Directive 6. |
| [`project-context/STATE.md`](project-context/STATE.md) | The resume point. Read first, every session. |
| [`project-context/`](project-context/) | Charter, requirements, feature register, generator and harness specs, tickets. |
| [`docs/results/`](docs/results/) | Generated result tables — the cost sweep, the cycle-4 verdict. |
| [`docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md`](docs/PRE-REGISTRATION-CYCLE4-2026-09-01.md) | What cycle 4 committed to, before it ran. |
| [`configs/rung_roster.yaml`](configs/rung_roster.yaml) | Every rung's status and citation — including the ones with no ladder row. |
| [`configs/journey.yaml`](configs/journey.yaml) | The three generations as cited literals. |
| `EVAL-LOCK*.json` | The lock chain. Never hand-edited. |
