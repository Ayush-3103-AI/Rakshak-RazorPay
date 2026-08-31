"""The results table generator (10-eval-harness-spec.md §8; T-152).

`make report` renders `docs/results_v2.md` and `docs/results_v2.parquet` from `EvalResult`
rows. This module reads **nothing else**: no panel, no labels, no event store, and above
all no split. It cannot open the test split because it never opens data at all — the only
input is rows another process already computed.

Three rules shape everything below.

**Provenance is load-bearing, not decoration.** Every rendered report carries the eval lock
sha, the open counter, the git sha, the scenario-config hash and the seeds, plus the live
result of ``lock.verify_lock()`` against the tree it is rendering from. If two rows disagree
about which lock or which commit produced them, the report **refuses to render**: a table
that blends two harnesses is worse than no table, because it looks like one.

**Prevalence is on every row, always (FR-021).** v1's headline was computed at 20%
prevalence against a real rate near 1.5% and reported without saying so. That single
omission is why v2 exists, so the column is not optional and there is no code path that
prints a PR-AUC without it.

**A rung that fails a floor is unmissable.** FLOOR-FAIL appears in a banner above the main
table, in the rung's own name, on the savings cell, in a dedicated verdict column, and
again in §7. Savings below the ``all_pass`` floor means the system costs more than doing
nothing, and no amount of ranking quality redeems that.

Negative results render in the same table, in the same style, sorted by rung and not by
score (Prime Directive 6). There is no failures appendix.

``report.py`` is deliberately absent from ``EVAL-LOCK.json``'s ``eval_modules``: it renders,
it does not compute a metric. That is why this file could be added after the freeze.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

import polars as pl

from rakshak.eval.lock import Drift, LockMismatchError, load_lock, verify_lock
from rakshak.schemas import EvalResult, TypologyId

__all__ = [
    "REPORT_MD",
    "REPORT_PARQUET",
    "Provenance",
    "ProvenanceError",
    "provenance_of",
    "read_result_dir",
    "read_results",
    "render",
    "to_frame",
    "write_report",
    "write_results",
]

#: Repo root, derived from this file rather than the working directory — `make report` run
#: from a subdirectory must still verify the same tree the lock hashes.
REPO_ROOT: Final = Path(__file__).resolve().parents[3]

REPORT_MD: Final = Path("docs/results_v2.md")
REPORT_PARQUET: Final = Path("docs/results_v2.parquet")
FIGURES_DIR: Final = Path("docs/figures")

#: The fields that are not plain float64. Everything else on ``EvalResult`` is, so the
#: schema survives a field being added without this file being edited.
_NON_FLOAT: Final[dict[str, pl.DataType]] = {
    "rung": pl.Int64(),
    "split": pl.String(),
    "cost_scenario": pl.String(),
    "recall_by_typology": pl.Struct({t.value: pl.Float64() for t in TypologyId}),
    "floor_fail": pl.List(pl.String()),
    "eval_lock_sha": pl.String(),
    "open_count": pl.Int64(),
    "git_sha": pl.String(),
}

SCHEMA: Final[dict[str, pl.DataType]] = {
    f.name: _NON_FLOAT.get(f.name, pl.Float64()) for f in fields(EvalResult)
}

#: The row-carried provenance. Disagreement on any of these is a refusal, not a warning.
_PROVENANCE_FIELDS: Final = ("eval_lock_sha", "open_count", "git_sha")


class ProvenanceError(RuntimeError):
    """Rows disagree about the harness that produced them, so no table is rendered."""


# ─────────────────────────────────────────────────────────────────────────────
# Rows in, rows out
# ─────────────────────────────────────────────────────────────────────────────


def to_frame(rows: Sequence[EvalResult]) -> pl.DataFrame:
    """`EvalResult` rows as a frame with an explicit schema.

    The schema is explicit because an empty ``floor_fail`` on every row would otherwise be
    inferred as ``List(Null)``, and a results file whose dtypes depend on whether anything
    failed is not a stable artifact.
    """
    payload = [
        {
            name: (
                {t.value: row.recall_by_typology.get(t, float("nan")) for t in TypologyId}
                if name == "recall_by_typology"
                else list(row.floor_fail)
                if name == "floor_fail"
                else str(getattr(row, name))
                if name in ("split", "cost_scenario")
                else getattr(row, name)
            )
            for name in SCHEMA
        }
        for row in rows
    ]
    return pl.DataFrame(payload, schema=SCHEMA)


def write_results(rows: Sequence[EvalResult], path: Path) -> Path:
    """Write ``docs/results_v2.parquet``. The rendered markdown is a view of this file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_frame(rows).write_parquet(path)
    return path


def _row_from_mapping(record: Mapping[str, Any]) -> EvalResult:
    """One ``EvalResult`` from a mapping, ignoring keys that are not fields.

    Extra keys are tolerated on purpose: ``cli.py`` writes its per-rung JSON with a dozen
    diagnostic keys alongside the row, and a reader that choked on them would force the
    scoring path to keep a second, poorer serialisation just for this module.
    """
    names = {f.name for f in fields(EvalResult)}
    missing = names - set(record) - {"cost_scenario", "floor_fail"}
    if missing:
        raise ValueError(f"not an EvalResult: missing {sorted(missing)}")
    kwargs = {k: v for k, v in record.items() if k in names}
    kwargs["recall_by_typology"] = {
        TypologyId(str(k)): float(v)
        for k, v in dict(kwargs.get("recall_by_typology") or {}).items()
    }
    kwargs["floor_fail"] = list(kwargs.get("floor_fail") or [])
    return EvalResult(**kwargs)


def read_results(path: Path) -> list[EvalResult]:
    """Read ``docs/results_v2.parquet`` back into rows."""
    return [_row_from_mapping(r) for r in pl.read_parquet(path).iter_rows(named=True)]


def read_result_dir(directory: Path) -> list[EvalResult]:
    """Every ``*.json`` row the scoring path wrote, sorted by (rung, split, scenario).

    ``cli.py`` writes one JSON per scored rung under ``data/v2/results/``; this is the
    bridge from those to the parquet, so that `make report` needs no change to the scoring
    path. ``data/`` is gitignored, so the parquet under ``docs/`` remains the artifact.
    """
    rows = [
        _row_from_mapping(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(directory.glob("*.json"))
    ]
    return _sorted(rows)


def _sorted(rows: Iterable[EvalResult]) -> list[EvalResult]:
    """By rung, then split, then scenario. Never by score — a loser does not sink."""
    return sorted(rows, key=lambda r: (r.rung, r.split, r.cost_scenario))


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Provenance:
    """The header, and the reason the report is allowed to exist at all."""

    eval_lock_sha: str
    open_count: int
    git_sha: str
    scenario_config_sha256: str
    frozen_at_git_sha: str
    seeds: tuple[int, ...]
    capacity_k: int
    drift: tuple[Drift, ...]
    #: The message from an enforced hash mismatch, or None. Never silently dropped.
    lock_error: str | None


def provenance_of(
    rows: Sequence[EvalResult], *, root: Path = REPO_ROOT, lock_path: Path | None = None
) -> Provenance:
    """The one provenance every row agrees on, or ``ProvenanceError``.

    ``eval_lock_sha``, ``open_count`` and ``git_sha`` are carried per row and must agree.
    The scenario-config hash and the seeds are read from ``EVAL-LOCK.json``, which is a
    single file and therefore cannot disagree with itself.

    ``verify_lock`` is called here rather than by the caller so that no rendering path can
    skip it. An enforced mismatch is caught and reported on the face of the report instead
    of raised: a reader who is handed nothing learns nothing, and a reader who is handed a
    table headed LOCK MISMATCH learns exactly the thing that matters.
    """
    if not rows:
        raise ValueError(
            "no EvalResult rows to render. `make report` reports what was measured; it "
            "does not invent a table. If you expected test-split rows, they do not exist "
            "yet — the test split opens exactly once, in T-151."
        )
    for name in _PROVENANCE_FIELDS:
        seen = {getattr(r, name) for r in rows}
        if len(seen) > 1:
            raise ProvenanceError(
                f"rows disagree on {name}: {sorted(map(str, seen))}. These rows were not "
                "produced by one harness, and a results table that blends two of them is "
                "worse than no table because it looks like one. Re-score every rung "
                "against the current lock, or render them as separate reports."
            )

    lock = load_lock(root, lock_path=lock_path)
    drift: list[Drift] = []
    lock_error: str | None = None
    try:
        drift = verify_lock(root, lock_path=lock_path)
    except LockMismatchError as exc:
        lock_error = str(exc)

    return Provenance(
        eval_lock_sha=rows[0].eval_lock_sha,
        open_count=rows[0].open_count,
        git_sha=rows[0].git_sha,
        scenario_config_sha256=str(lock.get("scenario_config_sha256", "unknown")),
        frozen_at_git_sha=str(lock.get("frozen_at_git_sha", "unknown")),
        seeds=tuple(lock.get("seeds", [])),
        capacity_k=int(lock.get("capacity_k", 0)),
        drift=tuple(drift),
        lock_error=lock_error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────


def _f(x: float, nd: int = 4) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:.2f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def _rung_name(row: EvalResult) -> str:
    tag = " · **FLOOR-FAIL**" if row.floor_fail else ""
    return f"**Rung {row.rung}**{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────


def _header(rows: Sequence[EvalResult], prov: Provenance) -> str:
    splits = sorted({r.split for r in rows})
    prevalences = sorted({r.prevalence for r in rows})
    lines = [
        "## 1. Provenance",
        "",
        "Every number below traces to exactly this commit, this lock, and this open count.",
        "A results table you cannot trace back to those is a results table nobody should",
        "believe — including you, in three days, when a number moves.",
        "",
        _table(
            ("field", "value"),
            [
                ("`git_sha` (rows)", f"`{prov.git_sha}`"),
                ("`eval_lock_sha`", f"`{prov.eval_lock_sha}`"),
                ("`open_count`", f"**{prov.open_count}**"),
                ("`scenario_config_sha256`", f"`{prov.scenario_config_sha256}`"),
                ("lock frozen at", f"`{prov.frozen_at_git_sha}`"),
                ("seeds", ", ".join(str(s) for s in prov.seeds) or "n/a"),
                ("capacity K", f"{prov.capacity_k} reviews/day"),
                ("splits rendered", ", ".join(splits)),
                ("prevalence", ", ".join(_pct(p) for p in prevalences)),
                ("rows", str(len(rows))),
            ],
        ),
        "",
    ]

    if prov.lock_error:
        lines += [
            "> ## LOCK MISMATCH — READ THIS BEFORE ANY NUMBER BELOW",
            ">",
            "> An enforced hash in `EVAL-LOCK.json` no longer matches the tree this report",
            "> was rendered from. Every number below was computed against different code",
            "> than the lock records, and none of it is comparable to anything measured",
            "> before the change.",
            ">",
            f"> `{prov.lock_error}`",
            "",
        ]
    elif prov.drift:
        lines += [
            "**Lock: enforced hashes verified. Unenforced drift recorded below** — these",
            "hashes are provenance of what existed at freeze time, not gates (see",
            "`EVAL-LOCK.json` `enforcement_note`).",
            "",
            *[
                f"- `{d.key}`: lock has `{d.expected[:16]}…`, tree has `{d.actual[:16]}…`"
                for d in prov.drift
            ],
            "",
        ]
    else:
        lines += [
            "**Lock: VERIFIED.** Every recorded hash matches the tree this was rendered",
            "from, so the harness that produced these numbers is the harness that was",
            "frozen before any v2 model existed.",
            "",
        ]

    if prov.open_count == 0 and any(r.split == "test" for r in rows):
        lines += [
            "> **PROVENANCE CONTRADICTION.** These rows include the test split while",
            "> `open_count` is 0. Either the open was never recorded in `EVAL-LOCK.json`",
            "> or these rows did not come from an authorised run. Do not publish this",
            "> table until that is resolved.",
            "",
        ]
    elif prov.open_count == 0:
        lines += [
            "`open_count` is **0**: the test split has never been opened. Everything below",
            "is measured on train/validation, as it must be until every rung is final.",
            "",
        ]
    return "\n".join(lines)


def _floor_fail_banner(rows: Sequence[EvalResult]) -> str:
    failed = [r for r in rows if r.floor_fail]
    if not failed:
        return (
            "**No rung fails a floor.** Every rung below beats `all_pass`, `all_hold`, "
            "`random_at_K` and `volume_rank` on savings.\n"
        )
    lines = [
        f"> # FLOOR-FAIL — {len(failed)} of {len(rows)} rows",
        ">",
        "> Savings below the `all_pass` floor means the system **costs more than doing "
        "nothing at all**.",
        "> No amount of ranking quality redeems that: a rung here is not a candidate for",
        "> adoption, whatever its PR-AUC says.",
        ">",
    ]
    lines += [
        f"> - **Rung {r.rung}** ({r.split}, `{r.cost_scenario}`) loses to: "
        f"**{', '.join(r.floor_fail)}** — savings {_f(r.savings)}"
        for r in failed
    ]
    lines.append("")
    return "\n".join(lines)


def _main_table(rows: Sequence[EvalResult]) -> str:
    base = [r for r in rows if r.cost_scenario == "base"] or list(rows)
    headline = _table(
        (
            "rung",
            "split",
            "prevalence",
            "PR-AUC",
            "savings",
            "floor all_pass",
            "floor all_hold",
            "floor random@K",
            "floor volume_rank",
            "gap to oracle",
            "median TTD (d)",
            "det@30d",
            "P@K",
            "alerts/day",
            "p99 ms",
            "verdict",
        ),
        [
            (
                _rung_name(r),
                r.split,
                _pct(r.prevalence),
                _f(r.pr_auc),
                f"**{_f(r.savings)} — FLOOR-FAIL**" if r.floor_fail else _f(r.savings),
                _f(r.savings_floor_all_pass),
                _f(r.savings_floor_all_hold),
                _f(r.savings_floor_random),
                _f(r.savings_floor_volume_rank),
                _pct(r.gap_to_oracle),
                _f(r.ttd_median_days, 1),
                _pct(r.detection_rate_d30),
                _f(r.precision_at_k),
                _f(r.alerts_per_day, 1),
                _f(r.p99_latency_ms, 3),
                "**FLOOR-FAIL**" if r.floor_fail else "ok",
            )
            for r in _sorted(base)
        ],
    )
    secondary = _table(
        (
            "rung",
            "split",
            "prevalence",
            "ROC-AUC",
            "ECE",
            "R@K",
            "det@7d",
            "det@14d",
            "alert Jaccard w/w",
            "state B p99",
            "model MB",
        ),
        [
            (
                _rung_name(r),
                r.split,
                _pct(r.prevalence),
                _f(r.roc_auc),
                _f(r.ece),
                _f(r.recall_at_k),
                _pct(r.detection_rate_d7),
                _pct(r.detection_rate_d14),
                _f(r.alert_jaccard_wow),
                _f(r.state_bytes_p99, 0),
                _f(r.model_size_mb, 3),
            )
            for r in _sorted(base)
        ],
    )
    return "\n".join(
        [
            "## 2. Main results",
            "",
            "One row per rung, at the base cost scenario. **Prevalence is on every row**",
            "(FR-021): a PR-AUC without the prevalence it was measured at is the exact",
            "mistake that made v1's headline meaningless. Rows are ordered by rung, not by",
            "score — a rung that lost sits where it belongs, in the same table and the same",
            "style as one that won (Prime Directive 6).",
            "",
            headline,
            "",
            "`alert Jaccard w/w` target is >= 0.60 (NFR-09); `det@Nd` is the share of",
            "uncensored positives detected within N days of `drift_onset_at`.",
            "",
            secondary,
            "",
        ]
    )


def _typology_table(rows: Sequence[EvalResult]) -> str:
    base = [r for r in rows if r.cost_scenario == "base"] or list(rows)
    order = list(TypologyId)
    body = []
    for r in _sorted(base):
        cells = []
        for t in order:
            v = r.recall_by_typology.get(t, float("nan"))
            # A typology at zero recall is the v1 R2 failure repeating, so it is marked
            # rather than left as a number the eye slides over.
            cells.append("**0.0000**" if (not math.isnan(v) and v <= 0.0) else _f(v))
        body.append((_rung_name(r), r.split, _pct(r.prevalence), *cells))
    return "\n".join(
        [
            "## 3. Per-typology recall",
            "",
            "Fraud is never one undifferentiated class. A rung that wins on the average",
            "while missing R2 entirely is visible here as exactly that — v1's slow-ramp",
            "failure was invisible in an aggregate and is the reason this table is on the",
            "front page rather than discoverable by someone who digs. Zero recall is bolded.",
            "",
            _table(
                ("rung", "split", "prevalence", *[t.value for t in order]),
                body,
            ),
            "",
        ]
    )


def _sweep(rows: Sequence[EvalResult]) -> str:
    scenarios = sorted({r.cost_scenario for r in rows})
    lines = [
        "## 4. Cost-asymmetry sweep",
        "",
        "v1 measured the false-hold/fraud-loss asymmetry at 47.5 / 13.1 / 61,368 against a",
        "literature band of 400-600. Three orders of magnitude of spread means the ratio",
        "cannot be assumed, so the ranking is reported at every ratio. **A ranking stable",
        "across the sweep is a far stronger claim than a win at one guessed ratio — and if",
        "it flips, that is the finding.**",
        "",
    ]
    if len(scenarios) < 2:
        lines += [
            f"**The sweep was not run.** Only the `{scenarios[0]}` cost scenario is present",
            "in `docs/results_v2.parquet`, so the ranking's stability under cost asymmetry",
            "is unknown and is not claimed. 10-eval-harness-spec.md §2 calls the sweep",
            "required, not optional; this is a gap in the results, not in the report.",
            "",
        ]
        return "\n".join(lines)

    # Stability is a claim about *order*, so it is judged on the rungs every scenario
    # scored. A rung missing from one scenario would otherwise read as a flip, which is a
    # finding this project must not manufacture.
    per_scenario = {s: [r for r in rows if r.cost_scenario == s] for s in scenarios}
    common = set.intersection(*({r.rung for r in v} for v in per_scenario.values()))

    rankings: dict[str, tuple[int, ...]] = {}
    body = []
    for s in scenarios:
        subset = sorted(
            per_scenario[s],
            key=lambda r: (-r.savings if not math.isnan(r.savings) else math.inf, r.rung),
        )
        rankings[s] = tuple(r.rung for r in subset if r.rung in common)
        body.append(
            (
                f"`{s}`",
                " > ".join(
                    f"**{r.rung}†**" if r.floor_fail else str(r.rung) for r in subset
                ),
                f"Rung {subset[0].rung}" if subset else "n/a",
                _f(subset[0].savings) if subset else "n/a",
            )
        )
    lines += [
        _table(("cost scenario", "ranking by savings (best first)", "winner", "savings"), body),
        "",
        "† = FLOOR-FAIL at that scenario: ranked, but not adoptable.",
        "",
    ]
    incomplete = sorted({r.rung for r in rows} - common)
    if incomplete:
        lines += [
            f"Rungs {incomplete} were not scored at every ratio and are excluded from the",
            "stability judgement below — a rung missing from one scenario is a gap in the",
            "sweep, not a flip in the ranking.",
            "",
        ]
    distinct = set(rankings.values())
    if len(distinct) == 1:
        lines += [
            "**The ranking is stable across the whole sweep.** The ordering does not depend",
            "on the cost ratio, which is the stronger of the two possible findings here.",
            "",
        ]
    else:
        lines += [
            f"**THE RANKING FLIPS.** {len(distinct)} distinct orderings across "
            f"{len(scenarios)} cost scenarios. Which rung wins depends on a ratio this",
            "project cannot measure to better than three orders of magnitude, so no single",
            "winner should be reported without its ratio attached.",
            "",
        ]
    return "\n".join(lines)


def _figures(root: Path) -> str:
    figures = sorted((root / FIGURES_DIR).glob("*.png"))
    lines = [
        "## 5. The G5 figure",
        "",
        "Alert rate against simulation day at `prevalence = 0`, with the six confounder",
        "windows shaded: raw features versus cohort residuals. Every alert on this figure",
        "is a false positive by construction. If the raw line spikes inside the bands and",
        "the residual line stays flat, that is the whole thesis of v2 in one image. If the",
        "lines are the same, charter K-1 has fired and the figure is published anyway.",
        "",
    ]
    if not figures:
        lines += [
            "**NOT PRODUCED.** No figure exists under `docs/figures/`. The eval spec §7",
            "calls this the single most valuable artifact of the sprint, so its absence is",
            "reported here rather than left as a missing image someone notices later.",
            "",
        ]
    else:
        lines += [
            *[f"![{p.stem}]({FIGURES_DIR.name}/{p.name})" for p in figures],
            "",
        ]
    return "\n".join(lines)


def _ablations(rows: Sequence[EvalResult], root: Path, lock_path: Path | None) -> str:
    """Rung-to-rung deltas against the margins declared before any v2 model existed."""
    lock = load_lock(root, lock_path=lock_path)
    margins = dict(lock.get("declared_adoption_margins", {}))
    rel_margin = float(margins.get("relative_pr_auc", 0.10))
    ttd_margin = float(margins.get("ttd_days", 3.0))

    base = {
        (r.rung, r.split): r for r in rows if r.cost_scenario == "base"
    } or {(r.rung, r.split): r for r in rows}
    body = []
    for (rung, split), row in sorted(base.items()):
        prev = base.get((rung - 1, split))
        if prev is None:
            continue
        rel = (
            (row.pr_auc - prev.pr_auc) / prev.pr_auc
            if prev.pr_auc not in (0.0,) and not math.isnan(prev.pr_auc)
            else float("nan")
        )
        ttd_gain = prev.ttd_median_days - row.ttd_median_days
        adopted = (not math.isnan(rel) and rel >= rel_margin) or (
            not math.isnan(ttd_gain) and ttd_gain >= ttd_margin
        )
        verdict = "**ADOPTED**" if adopted else "**NOT ADOPTED — negative result**"
        if row.floor_fail:
            verdict = "**NOT ADOPTED — FLOOR-FAIL**"
        body.append(
            (
                f"Rung {rung} vs {rung - 1}",
                split,
                _pct(row.prevalence),
                _f(prev.pr_auc),
                _f(row.pr_auc),
                _pct(rel),
                _f(ttd_gain, 1),
                verdict,
            )
        )

    lines = [
        "## 6. Ablations and adoption verdicts",
        "",
        f"A rung is adopted only if it beats the rung below by **>= {rel_margin:.0%} relative",
        f"PR-AUC** or reduces median TTD by **>= {ttd_margin:.0f} days**, at equal alerts per",
        "analyst-day. Those margins were declared in `EVAL-LOCK.json` before any v2 model",
        "existed; they are read from the lock here, not restated by hand.",
        "",
    ]
    if body:
        lines += [
            _table(
                (
                    "comparison",
                    "split",
                    "prevalence",
                    "PR-AUC below",
                    "PR-AUC above",
                    "relative delta",
                    "TTD gain (d)",
                    "verdict",
                ),
                body,
            ),
            "",
            "**Rung 3 vs Rung 2 is the K-1 test** — the cohort-residual hypothesis on and",
            "off. If that relative delta is under 5%, K-1 has fired: it is written up in",
            "`LIMITATIONS.md` with the number, and no features are added to rescue it. A",
            "clean falsification of a well-motivated hypothesis, on a harness frozen in",
            "advance, is a real result.",
            "",
        ]
    else:
        lines += ["Fewer than two consecutive rungs are present; no comparison is possible.\n"]

    lines += [
        "Three ablations named in 10-eval-harness-spec.md §8.6 are **not rendered**:",
        "leave-one-family-out, T1 vs T1+T2, and the graph-feature re-test. `EvalResult` has",
        "no field that distinguishes an ablation variant from a rung, so they cannot be",
        "carried in `docs/results_v2.parquet` as it is specified in 09-interfaces.md §11.",
        "Naming the gap here beats rendering three empty tables.",
        "",
    ]
    return "\n".join(lines)


def _limitations(rows: Sequence[EvalResult]) -> str:
    failed = [r for r in rows if r.floor_fail]
    lines = [
        "## 7. Limitations",
        "",
        "Full text: [`LIMITATIONS.md`](../LIMITATIONS.md). Everything that did not work is",
        "on this page and in that file, with its number. Nothing is in an appendix.",
        "",
    ]
    if failed:
        lines += [
            "Failed rungs on this run, with their numbers:",
            "",
            *[
                f"- **Rung {r.rung}** ({r.split}, `{r.cost_scenario}`, prevalence "
                f"{_pct(r.prevalence)}): savings {_f(r.savings)}, below "
                f"{', '.join(r.floor_fail)}. PR-AUC {_f(r.pr_auc)} does not redeem it."
                for r in failed
            ],
            "",
        ]
    else:
        lines += ["No rung on this run fell below a floor.\n"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────────────────────


def render(
    rows: Sequence[EvalResult], *, root: Path = REPO_ROOT, lock_path: Path | None = None
) -> str:
    """The whole report as markdown. Refuses if the rows do not share one provenance."""
    ordered = _sorted(rows)
    prov = provenance_of(ordered, root=root, lock_path=lock_path)
    return "\n".join(
        [
            "# Rakshak v2 — Results",
            "",
            "> Post-onboarding merchant risk sentinel. Generated by `make report` from",
            "> `docs/results_v2.parquet`. Do not hand-edit: the next run overwrites it.",
            "",
            _header(ordered, prov),
            _floor_fail_banner(ordered),
            _main_table(ordered),
            _typology_table(ordered),
            _sweep(ordered),
            _figures(root),
            _ablations(ordered, root, lock_path),
            _limitations(ordered),
        ]
    )


def write_report(
    rows: Sequence[EvalResult], *, root: Path = REPO_ROOT, lock_path: Path | None = None
) -> Path:
    """Render to ``docs/results_v2.md`` and write ``docs/results_v2.parquet`` beside it.

    The markdown is written last, so a refusal on provenance leaves no half-written report.
    """
    text = render(rows, root=root, lock_path=lock_path)
    write_results(_sorted(rows), root / REPORT_PARQUET)
    target = root / REPORT_MD
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
