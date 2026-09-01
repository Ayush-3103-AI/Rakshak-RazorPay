"""T-152a — the results table generator.

No rung is trained yet, so every row here is synthetic and built in this file. That is not
a placeholder: the report must be correct about provenance, prevalence and FLOOR-FAIL
before any real number exists, exactly as the metric suite was.

Nothing here touches the real ``EVAL-LOCK.json`` or the real ``docs/``. Every test renders
against a repo-shaped tree in ``tmp_path`` with its own lock, so a test cannot rewrite the
one-way door and cannot overwrite a published report.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import polars as pl
import pytest

from rakshak.eval.lock import EVAL_MODULES, write_lock
from rakshak.eval.report import (
    ProvenanceError,
    provenance_of,
    read_result_dir,
    read_results,
    render,
    to_frame,
    write_report,
)
from rakshak.schemas import EvalResult, TypologyId

LOCK_SHA = "0" * 64
GIT_SHA = "b4bb2ab1d6eee2d1f836f23da162c3221a52cc07"


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A repo-shaped tree with its own lock, written the way T-133 writes the real one."""
    for rel in EVAL_MODULES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# stand-in for {rel}\nVALUE = 1\n", encoding="utf-8")
    gen = tmp_path / "src" / "rakshak" / "generator"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "engine.py").write_text("# generator\n", encoding="utf-8")
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "scenario_v2.yaml").write_text("seed: 42\n", encoding="utf-8")
    write_lock(tmp_path)
    return tmp_path


def make_row(
    *,
    rung: int = 2,
    split: str = "val",
    prevalence: float = 0.015,
    pr_auc: float = 0.42,
    savings: float = 0.31,
    floors: tuple[float, float, float, float] = (0.10, -0.40, 0.05, 0.12),
    ttd: float = 9.0,
    recall: dict[TypologyId, float] | None = None,
    cost_scenario: str = "base",
    floor_fail: list[str] | None = None,
    eval_lock_sha: str = LOCK_SHA,
    open_count: int = 0,
    git_sha: str = GIT_SHA,
) -> EvalResult:
    """A synthetic ``EvalResult``. ``floor_fail`` mirrors ``Floors.failed_by`` by default."""
    all_pass, all_hold, random_at_k, volume_rank = floors
    if floor_fail is None:
        floor_fail = [
            name
            for name, floor in zip(
                ("all_pass", "all_hold", "random_at_k", "volume_rank"), floors, strict=True
            )
            if savings <= floor
        ]
    return EvalResult(
        rung=rung,
        split=split,  # type: ignore[arg-type]
        prevalence=prevalence,
        pr_auc=pr_auc,
        roc_auc=0.88,
        ece=0.031,
        savings=savings,
        savings_floor_random=random_at_k,
        savings_floor_all_pass=all_pass,
        savings_floor_all_hold=all_hold,
        savings_floor_volume_rank=volume_rank,
        precision_at_k=0.24,
        recall_at_k=0.19,
        alerts_per_day=50.0,
        ttd_median_days=ttd,
        detection_rate_d7=0.21,
        detection_rate_d14=0.44,
        detection_rate_d30=0.66,
        gap_to_oracle=0.38,
        alert_jaccard_wow=0.71,
        recall_by_typology=recall
        or {t: 0.5 for t in TypologyId},
        p99_latency_ms=4.2,
        state_bytes_p99=7091.0,
        model_size_mb=1.4,
        eval_lock_sha=eval_lock_sha,
        open_count=open_count,
        git_sha=git_sha,
        cost_scenario=cost_scenario,
        floor_fail=floor_fail,
    )


# ───────────────────────────── round trip ─────────────────────────────


def test_frame_schema_is_stable_when_nothing_failed() -> None:
    """``floor_fail`` stays List(String) even when every list is empty."""
    frame = to_frame([make_row(savings=0.9)])
    assert frame.schema["floor_fail"] == pl.List(pl.String)
    assert frame.schema["recall_by_typology"] == pl.Struct(
        {t.value: pl.Float64 for t in TypologyId}
    )
    assert frame.schema["prevalence"] == pl.Float64


def test_parquet_round_trip_preserves_every_field(tmp_path: Path) -> None:
    rows = [make_row(rung=1, savings=0.02), make_row(rung=2, savings=0.31)]
    path = tmp_path / "results_v2.parquet"
    from rakshak.eval.report import write_results

    write_results(rows, path)
    back = read_results(path)
    assert back == rows


def test_read_result_dir_tolerates_the_cli_json_sidecars(tmp_path: Path) -> None:
    """``cli.py`` writes a dozen diagnostic keys beside the row; they are ignored.

    ``label`` is the one exception and it is asserted separately below: it stopped being a
    diagnostic and became a field, because a results table that cannot name its own rows
    renders three Rung-0 floors and both exposure arms of every rung as the same string.
    """
    row = make_row(rung=3)
    payload = {
        "rung": row.rung,
        "split": row.split,
        "prevalence": row.prevalence,
        "pr_auc": row.pr_auc,
        "roc_auc": row.roc_auc,
        "ece": row.ece,
        "savings": row.savings,
        "savings_floor_random": row.savings_floor_random,
        "savings_floor_all_pass": row.savings_floor_all_pass,
        "savings_floor_all_hold": row.savings_floor_all_hold,
        "savings_floor_volume_rank": row.savings_floor_volume_rank,
        "precision_at_k": row.precision_at_k,
        "recall_at_k": row.recall_at_k,
        "alerts_per_day": row.alerts_per_day,
        "ttd_median_days": row.ttd_median_days,
        "detection_rate_d7": row.detection_rate_d7,
        "detection_rate_d14": row.detection_rate_d14,
        "detection_rate_d30": row.detection_rate_d30,
        "gap_to_oracle": row.gap_to_oracle,
        "alert_jaccard_wow": row.alert_jaccard_wow,
        "recall_by_typology": {t.value: v for t, v in row.recall_by_typology.items()},
        "p99_latency_ms": row.p99_latency_ms,
        "state_bytes_p99": row.state_bytes_p99,
        "model_size_mb": row.model_size_mb,
        "eval_lock_sha": row.eval_lock_sha,
        "open_count": row.open_count,
        "git_sha": row.git_sha,
        "cost_scenario": row.cost_scenario,
        "floor_fail": row.floor_fail,
        # diagnostics cli.py adds — must not break the reader
        "label": "rung3",
        "capacity_k": 50,
        "beats_all_floors": True,
        "n_rows_scored": 12345,
    }
    (tmp_path / "rung3_val_seed42.json").write_text(json.dumps(payload), encoding="utf-8")
    [got] = read_result_dir(tmp_path)
    # Every diagnostic key is still ignored...
    assert dataclasses.replace(got, label="") == row
    # ...except `label`, which is now carried onto the row and rendered beside the number.
    assert got.label == "rung3"


def test_a_row_without_a_label_still_reads(tmp_path: Path) -> None:
    """`label` is defaulted, so a sidecar written before it existed still round-trips."""
    row = make_row(rung=2)
    payload = {f.name: getattr(row, f.name) for f in dataclasses.fields(row)}
    payload["recall_by_typology"] = {
        t.value: v for t, v in row.recall_by_typology.items()
    }
    payload["split"] = str(row.split)
    del payload["label"]
    (tmp_path / "rung2_val_seed42.json").write_text(json.dumps(payload), encoding="utf-8")
    [got] = read_result_dir(tmp_path)
    assert got.label == ""
    assert got == row


# ───────────────────────────── provenance ─────────────────────────────


def test_refuses_to_render_when_rows_disagree_on_the_lock(fake_root: Path) -> None:
    """Two locks in one table is the failure this check exists to prevent."""
    rows = [make_row(rung=1), make_row(rung=2, eval_lock_sha="f" * 64)]
    with pytest.raises(ProvenanceError, match="eval_lock_sha"):
        render(rows, root=fake_root)


def test_refuses_to_render_when_rows_disagree_on_open_count(fake_root: Path) -> None:
    rows = [make_row(rung=1, open_count=0), make_row(rung=2, open_count=1)]
    with pytest.raises(ProvenanceError, match="open_count"):
        render(rows, root=fake_root)


def test_refuses_to_render_when_rows_disagree_on_git_sha(fake_root: Path) -> None:
    rows = [make_row(rung=1), make_row(rung=2, git_sha="deadbeef")]
    with pytest.raises(ProvenanceError, match="git_sha"):
        render(rows, root=fake_root)


def test_refuses_to_render_rows_that_do_not_exist(fake_root: Path) -> None:
    with pytest.raises(ValueError, match="does not invent a table"):
        render([], root=fake_root)


def test_header_carries_every_provenance_field(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    lock = json.loads((fake_root / "EVAL-LOCK.json").read_text(encoding="utf-8"))
    assert LOCK_SHA in text
    assert GIT_SHA in text
    assert lock["scenario_config_sha256"] in text
    assert "`open_count`" in text
    assert "42" in text  # seeds
    assert "VERIFIED" in text


def test_a_drifted_tree_says_so_on_its_face(fake_root: Path) -> None:
    """An enforced hash mismatch is rendered, loudly, not swallowed and not raised."""
    (fake_root / EVAL_MODULES[0]).write_text("# edited after the lock\n", encoding="utf-8")
    text = render([make_row()], root=fake_root)
    assert "LOCK MISMATCH" in text
    assert "READ THIS BEFORE ANY NUMBER BELOW" in text
    assert text.index("LOCK MISMATCH") < text.index("## 2. Main results")


def test_unenforced_drift_is_reported_as_drift_not_as_a_pass(fake_root: Path) -> None:
    (fake_root / "src" / "rakshak" / "generator" / "engine.py").write_text(
        "# generator moved on\n", encoding="utf-8"
    )
    text = render([make_row()], root=fake_root)
    assert "generator_module_sha256" in text
    assert "LOCK MISMATCH" not in text


def test_provenance_of_reports_the_lock_error_without_raising(fake_root: Path) -> None:
    (fake_root / EVAL_MODULES[1]).write_text("# changed\n", encoding="utf-8")
    prov = provenance_of([make_row()], root=fake_root)
    assert prov.lock_error is not None
    assert "eval_module_sha256" in prov.lock_error


def test_test_rows_with_a_zero_open_count_are_flagged(fake_root: Path) -> None:
    """A test-split row against open_count 0 means the open was never recorded."""
    text = render([make_row(split="test", open_count=0)], root=fake_root)
    assert "PROVENANCE CONTRADICTION" in text


# ───────────────────────────── prevalence (FR-021) ─────────────────────────────


def test_prevalence_is_printed_on_every_row_of_every_table(fake_root: Path) -> None:
    rows = [make_row(rung=1, prevalence=0.015), make_row(rung=2, prevalence=0.015)]
    text = render(rows, root=fake_root)
    # header, main table, secondary table, per-typology table — one column each
    assert text.count("1.50%") >= 1 + 2 + 2 + 2
    assert "prevalence" in text


def test_no_pr_auc_table_row_without_its_prevalence(fake_root: Path) -> None:
    """The v1 original sin, as an assertion: a PR-AUC row always carries prevalence."""
    text = render([make_row(prevalence=0.2, pr_auc=0.9)], root=fake_root)
    for line in text.splitlines():
        if "0.9000" in line and line.startswith("|"):
            assert "20.00%" in line


# ───────────────────────────── FLOOR-FAIL ─────────────────────────────


def test_a_floor_failing_rung_is_impossible_to_miss(fake_root: Path) -> None:
    """Banner, rung name, savings cell, verdict column, and §7 — five places."""
    rows = [
        make_row(rung=1, savings=0.02),  # below all_pass 0.10 and volume_rank 0.12
        make_row(rung=2, savings=0.31),
    ]
    text = render(rows, root=fake_root)
    assert text.count("FLOOR-FAIL") >= 5
    banner = text.index("# FLOOR-FAIL")
    assert banner < text.index("## 2. Main results")
    assert "all_pass" in text[banner : text.index("## 2. Main results")]
    assert "costs more than doing nothing" in text
    # the losing rung is in the same table as the winner, ordered by rung not by score
    main = text[text.index("## 2. Main results") : text.index("## 3.")]
    assert main.index("**Rung 1**") < main.index("**Rung 2**")


def test_a_clean_run_says_no_rung_fails_a_floor(fake_root: Path) -> None:
    text = render([make_row(savings=0.9)], root=fake_root)
    assert "No rung fails a floor" in text
    assert "FLOOR-FAIL" not in text.split("## 2. Main results")[0].split("Three rules")[-1]


def test_floor_fail_survives_a_high_pr_auc(fake_root: Path) -> None:
    """Ranking quality does not redeem savings below the floor, and the text says so."""
    text = render([make_row(rung=2, pr_auc=0.99, savings=0.01)], root=fake_root)
    assert "does not redeem it" in text
    assert "NOT ADOPTED" not in text or "FLOOR-FAIL" in text


# ───────────────────────────── per-typology ─────────────────────────────


def test_per_typology_recall_gets_its_own_table_with_all_nine(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    section = text[text.index("## 3. Per-typology") : text.index("## 4.")]
    for t in TypologyId:
        assert t.value in section


def test_a_typology_at_zero_recall_is_bolded(fake_root: Path) -> None:
    """A rung that wins on the average while missing R2 must be visible as exactly that."""
    recall = {t: 0.6 for t in TypologyId}
    recall[TypologyId.R2] = 0.0
    text = render([make_row(pr_auc=0.8, recall=recall)], root=fake_root)
    section = text[text.index("## 3. Per-typology") : text.index("## 4.")]
    assert "**0.0000**" in section


# ───────────────────────────── sweep, figures, ablations ─────────────────────────────


def test_a_single_cost_scenario_reports_the_sweep_as_not_run(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    assert "The sweep was not run" in text


def test_a_stable_ranking_across_the_sweep_is_stated(fake_root: Path) -> None:
    rows = [
        make_row(rung=r, savings=s, cost_scenario=c)
        for c in ("ratio_0.01", "ratio_1", "ratio_100")
        for r, s in ((1, 0.20), (2, 0.50))
    ]
    text = render(rows, root=fake_root)
    assert "ranking is stable" in text


def test_a_flipping_ranking_is_reported_as_the_finding(fake_root: Path) -> None:
    rows = [
        make_row(rung=1, savings=0.6, cost_scenario="ratio_0.01"),
        make_row(rung=2, savings=0.2, cost_scenario="ratio_0.01"),
        make_row(rung=1, savings=0.2, cost_scenario="ratio_100"),
        make_row(rung=2, savings=0.6, cost_scenario="ratio_100"),
    ]
    text = render(rows, root=fake_root)
    assert "THE RANKING FLIPS" in text


def test_a_missing_g5_figure_is_reported_not_hidden(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    assert "NOT PRODUCED" in text


def test_a_present_figure_is_linked(fake_root: Path) -> None:
    figures = fake_root / "docs" / "figures"
    figures.mkdir(parents=True)
    (figures / "g5_confounder_null.png").write_bytes(b"\x89PNG")
    text = render([make_row()], root=fake_root)
    assert "![g5_confounder_null](figures/g5_confounder_null.png)" in text


def test_adoption_uses_the_margins_declared_in_the_lock(fake_root: Path) -> None:
    """Rung 3 vs Rung 2 under the 10% margin: 4% relative is a negative result."""
    rows = [
        make_row(rung=2, pr_auc=0.50, savings=0.5, ttd=9.0),
        make_row(rung=3, pr_auc=0.52, savings=0.5, ttd=9.0),
    ]
    text = render(rows, root=fake_root)
    section = text[text.index("## 6. Ablations") :]
    assert "NOT ADOPTED — negative result" in section
    assert "4.00%" in section
    assert "K-1" in section


def test_a_rung_that_clears_the_margin_is_adopted(fake_root: Path) -> None:
    rows = [
        make_row(rung=2, pr_auc=0.40, savings=0.5),
        make_row(rung=3, pr_auc=0.60, savings=0.5),
    ]
    text = render(rows, root=fake_root)
    assert "**ADOPTED**" in text


def test_ttd_alone_can_carry_adoption(fake_root: Path) -> None:
    """Charter §2 makes TTD an equal-standing win condition, not a tiebreaker."""
    rows = [
        make_row(rung=2, pr_auc=0.50, ttd=12.0, savings=0.5),
        make_row(rung=3, pr_auc=0.50, ttd=8.0, savings=0.5),
    ]
    text = render(rows, root=fake_root)
    assert "**ADOPTED**" in text


def test_unrenderable_ablations_are_named_rather_than_faked(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    assert "leave-one-family-out" in text
    assert "no field that distinguishes an ablation variant" in text


# ───────────────────────────── the whole artifact ─────────────────────────────


def test_write_report_writes_both_artifacts_and_is_deterministic(fake_root: Path) -> None:
    rows = [make_row(rung=1, savings=0.02), make_row(rung=2, savings=0.31)]
    target = write_report(rows, root=fake_root)
    assert target == fake_root / "docs" / "results_v2.md"
    first = target.read_text(encoding="utf-8")
    parquet = fake_root / "docs" / "results_v2.parquet"
    assert read_results(parquet) == rows
    # no timestamp in the output: re-rendering the same rows must not produce a diff
    write_report(rows, root=fake_root)
    assert target.read_text(encoding="utf-8") == first


def test_the_sections_appear_in_the_order_the_spec_names(fake_root: Path) -> None:
    text = render([make_row()], root=fake_root)
    order = [
        "## 1. Provenance",
        "## 2. Main results",
        "## 3. Per-typology recall",
        "## 4. Cost-asymmetry sweep",
        "## 5. The G5 figure",
        "## 6. Ablations and adoption verdicts",
        "## 7. Limitations",
    ]
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions)
    assert "LIMITATIONS.md" in text


def test_nan_metrics_render_as_not_available_rather_than_zero(fake_root: Path) -> None:
    """An unmeasured latency is `n/a`; a flattering 0.0000 would be a lie."""
    row = make_row(ttd=float("nan"))
    assert math.isnan(row.ttd_median_days)
    text = render([row], root=fake_root)
    assert "n/a" in text


def test_a_refusal_leaves_no_half_written_report(fake_root: Path) -> None:
    rows = [make_row(rung=1), make_row(rung=2, open_count=99)]
    with pytest.raises(ProvenanceError):
        write_report(rows, root=fake_root)
    assert not (fake_root / "docs" / "results_v2.md").exists()


def test_a_rung_missing_from_one_ratio_is_not_reported_as_a_flip(fake_root: Path) -> None:
    """Membership is not order. A gap in the sweep must not manufacture a finding."""
    rows = [
        make_row(rung=1, savings=0.2, cost_scenario="ratio_0.01"),
        make_row(rung=2, savings=0.5, cost_scenario="ratio_0.01"),
        make_row(rung=0, savings=0.1, cost_scenario="ratio_100"),
        make_row(rung=1, savings=0.2, cost_scenario="ratio_100"),
        make_row(rung=2, savings=0.5, cost_scenario="ratio_100"),
    ]
    text = render(rows, root=fake_root)
    assert "THE RANKING FLIPS" not in text
    assert "ranking is stable" in text
    assert "not scored at every ratio" in text
