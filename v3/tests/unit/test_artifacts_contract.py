"""The artefact contract, tested against externally observable properties (T-0126, #60).

Every test here asserts something a reader of the emitted files could check for
themselves — a byte-identical rebuild, a refusal with a named reason, the absence of a
forbidden key — rather than an internal call sequence.

The result rows under ``data/v2/eval/`` are used as **structural** fixtures only. They are
stale cycle-1 numbers being regenerated at the corrected cycle-2 geometry as this is
written, so nothing here asserts on a value; what is asserted is that whatever the values
turn out to be, the contract holds over them.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from rakshak.artifacts import (
    FORBIDDEN_KEYS,
    SCHEMA_VERSION,
    SPLIT_VALUES,
    ArtifactSchemaError,
    canonical_bytes,
    envelope,
    sanitise,
    split_label,
    validate,
    validate_file,
)
from rakshak.artifacts.build import (
    REPO_ROOT,
    build_all,
    build_g5,
    build_ladder,
    build_lock_state,
    read_result_rows,
)
from rakshak.schemas import EvalResult

RESULTS_DIR = REPO_ROOT / "data" / "v2" / "eval"
has_results = pytest.mark.skipif(
    not (RESULTS_DIR.is_dir() and any(RESULTS_DIR.glob("*.json"))),
    reason="no scored results in the tree; the contract tests that need rows are skipped",
)


def _g5_fixture(n_days: int = 5) -> dict[str, Any]:
    """A minimal, structurally valid G5 gate dump.

    The real one does not exist yet — the gate records its verdicts to the terminal and
    dumps no series — so the contract is tested against the shape it requires rather than
    against numbers that would be invented here.
    """
    return {
        "prevalence": 0.0,
        "nominal_alert_rate": 0.005,
        "excess_allowed_pp": 2.0,
        "n_days": n_days,
        "windows": [
            {
                "confounder": "P1",
                "start_day": 1,
                "end_day": 3,
                "role": "control",
                "feature": "v_txn_count",
            },
            {
                "confounder": "P2",
                "start_day": 3,
                "end_day": 4,
                "role": "adversarial",
                "feature": "v_ticket_cv",
            },
        ],
        "series": [
            {
                "detector": "raw",
                "threshold": 2.9,
                "quiet_day_rate": 0.005,
                "alert_rate_by_day": [0.004] * n_days,
                "window_excess": [{"confounder": "P1", "alert_rate": 0.09, "excess_pp": 8.5}],
                "verdict": "RED",
            },
            {
                "detector": "cohort-residual",
                "threshold": 2.7,
                "quiet_day_rate": 0.005,
                "alert_rate_by_day": [0.005] * n_days,
                "window_excess": [{"confounder": "P1", "alert_rate": 0.006, "excess_pp": 0.1}],
                "verdict": "GREEN",
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation and determinism
# ─────────────────────────────────────────────────────────────────────────────


def test_regenerating_is_byte_identical(tmp_path: Path) -> None:
    """The acceptance criterion, as a test: same inputs, same bytes, twice."""
    first, second = tmp_path / "a", tmp_path / "b"
    build_all(REPO_ROOT, out_dir=first)
    build_all(REPO_ROOT, out_dir=second)

    names = sorted(p.name for p in first.glob("*.json"))
    assert names, "the generator emitted nothing at all"
    assert names == sorted(p.name for p in second.glob("*.json"))
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), (
            f"{name} differs between two runs over identical inputs — something in the "
            "serialisation reads the clock, the filesystem order, or a dict's insertion order"
        )


def test_no_wall_clock_stamp_in_any_emitted_file(tmp_path: Path) -> None:
    """No ``generated_at``. A timestamp is the cheapest way to break byte-identity."""
    build_all(REPO_ROOT, out_dir=tmp_path)
    for path in tmp_path.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert "generated_at" not in json.dumps(doc), f"{path.name} stamps a generation time"


def test_canonical_bytes_refuses_non_finite() -> None:
    """``NaN`` must never reach a file: no ``JSON.parse`` on earth accepts it."""
    with pytest.raises(ValueError):
        canonical_bytes({"x": float("nan")})


def test_sanitise_records_why_a_value_became_null() -> None:
    clean, found = sanitise({"ttd_median_days": float("inf"), "precision_at_k": float("nan")})
    assert clean == {"ttd_median_days": None, "precision_at_k": None}
    assert found == {"ttd_median_days": {"Infinity": 1}, "precision_at_k": {"NaN": 1}}
    canonical_bytes(clean)  # and the result is now serialisable


def test_canonical_bytes_is_key_order_independent() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})


# ─────────────────────────────────────────────────────────────────────────────
# The validator — every rejection names the artefact and the reason
# ─────────────────────────────────────────────────────────────────────────────


def _doc(**over: Any) -> dict[str, Any]:
    base = envelope(
        "ladder",
        {
            "rungs": [{"split": "VALIDATION", "rung": 2}],
            "capacity_k": 9,
            "metric_keys": ["pr_auc"],
        },
        split="VALIDATION",
        provenance={},
    )
    base.update(over)
    return base


def test_valid_document_passes() -> None:
    assert validate(_doc()) == "ladder"


def test_schema_version_mismatch_rejects_with_name_and_reason() -> None:
    with pytest.raises(ArtifactSchemaError) as exc:
        validate(_doc(schema_version="v2.9.9"))
    assert exc.value.artifact == "ladder"
    assert "schema_version" in exc.value.reason and SCHEMA_VERSION in exc.value.reason


def test_unknown_artifact_name_rejects() -> None:
    with pytest.raises(ArtifactSchemaError, match="unknown artifact name"):
        validate(_doc(artifact="something_else"))


def test_missing_payload_key_rejects() -> None:
    doc = _doc()
    del doc["payload"]["metric_keys"]
    with pytest.raises(ArtifactSchemaError, match="missing required key 'metric_keys'"):
        validate(doc)


def test_numeric_row_without_a_split_field_rejects() -> None:
    """The split is a field, not a filename convention — enforced, not documented."""
    doc = _doc()
    del doc["payload"]["rungs"][0]["split"]
    with pytest.raises(ArtifactSchemaError, match="carries its split as a field"):
        validate(doc)


def test_bad_split_value_rejects() -> None:
    doc = _doc()
    doc["payload"]["rungs"][0]["split"] = "holdout"
    with pytest.raises(ArtifactSchemaError, match="split 'holdout'"):
        validate(doc)


@pytest.mark.parametrize("field", sorted(FORBIDDEN_KEYS))
def test_every_ground_truth_field_is_rejected_at_any_depth(field: str) -> None:
    doc = _doc()
    doc["payload"]["rungs"][0]["nested"] = {"deeper": [{field: "leaked"}]}
    with pytest.raises(ArtifactSchemaError, match="ground-truth field"):
        validate(doc)


def test_split_label_refuses_an_unknown_split() -> None:
    assert split_label("val") == "VALIDATION"
    with pytest.raises(ArtifactSchemaError, match="unknown split"):
        split_label("holdout")


def test_unparseable_file_rejects_by_name(tmp_path: Path) -> None:
    bad = tmp_path / "ladder.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactSchemaError) as exc:
        validate_file(bad)
    assert exc.value.artifact == "ladder.json"


# ─────────────────────────────────────────────────────────────────────────────
# The lock artefact — N locks, discovered, chain resolved
# ─────────────────────────────────────────────────────────────────────────────


def test_lock_state_reports_every_lock_with_its_three_hashes(tmp_path: Path) -> None:
    payload, paths = build_lock_state(REPO_ROOT)
    assert len(payload["locks"]) == len(paths) >= 2
    for lock in payload["locks"]:
        assert set(lock["hashes"]) == {
            "eval_module_sha256",
            "generator_module_sha256",
            "scenario_config_sha256",
        }
        assert all(isinstance(v, str) and len(v) == 64 for v in lock["hashes"].values())
        assert isinstance(lock["open_count"], int)
        assert isinstance(lock["frozen_at_git_sha"], str) and lock["frozen_at_git_sha"]


def test_exactly_one_lock_is_authoritative_and_it_is_the_head_of_the_chain() -> None:
    payload, _ = build_lock_state(REPO_ROOT)
    live = [lock for lock in payload["locks"] if lock["authoritative"]]
    assert len(live) == 1
    assert live[0]["file"] == payload["authoritative_lock"]
    assert live[0]["superseded_by"] is None
    assert live[0]["cycle"] == max(lock["cycle"] for lock in payload["locks"])
    # And the superseded one names its successor, so the chain reads in both directions.
    superseded = [lock for lock in payload["locks"] if not lock["authoritative"]]
    assert all(lock["superseded_by"] for lock in superseded)


def test_lock_discovery_scales_to_n_locks(tmp_path: Path) -> None:
    """A third cycle must need no code change. Written as three synthetic locks."""

    def lock(cycle: int, supersedes: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "eval_module_sha256": "a" * 64,
            "generator_module_sha256": "b" * 64,
            "scenario_config_sha256": "c" * 64,
            "enforced": ["eval_module_sha256"],
            "open_count": 0,
            "open_log": [],
            "frozen_at_git_sha": f"sha-{cycle}",
        }
        if cycle > 1:
            body["cycle"] = cycle
            body["supersedes"] = supersedes
        return body

    (tmp_path / "EVAL-LOCK.json").write_text(json.dumps(lock(1, None)), encoding="utf-8")
    (tmp_path / "EVAL-LOCK-CYCLE2.json").write_text(
        json.dumps(lock(2, "EVAL-LOCK.json")), encoding="utf-8"
    )
    (tmp_path / "EVAL-LOCK-CYCLE3.json").write_text(
        json.dumps(lock(3, "EVAL-LOCK-CYCLE2.json")), encoding="utf-8"
    )
    payload, _ = build_lock_state(tmp_path)
    assert [lock_["cycle"] for lock_ in payload["locks"]] == [1, 2, 3]
    assert payload["authoritative_lock"] == "EVAL-LOCK-CYCLE3.json"
    assert payload["n_locks"] == 3


def test_two_unsuperseded_locks_is_a_refusal(tmp_path: Path) -> None:
    body = {
        "eval_module_sha256": "a" * 64,
        "generator_module_sha256": "b" * 64,
        "scenario_config_sha256": "c" * 64,
        "open_count": 0,
        "frozen_at_git_sha": "x",
    }
    (tmp_path / "EVAL-LOCK.json").write_text(json.dumps(body), encoding="utf-8")
    (tmp_path / "EVAL-LOCK-CYCLE2.json").write_text(
        json.dumps({**body, "cycle": 2}), encoding="utf-8"
    )
    with pytest.raises(ArtifactSchemaError, match="exactly one unsuperseded lock"):
        build_lock_state(tmp_path)


def test_a_broken_supersession_chain_is_a_refusal(tmp_path: Path) -> None:
    body = {
        "eval_module_sha256": "a" * 64,
        "generator_module_sha256": "b" * 64,
        "scenario_config_sha256": "c" * 64,
        "open_count": 0,
        "frozen_at_git_sha": "x",
        "cycle": 2,
        "supersedes": "EVAL-LOCK-THAT-IS-NOT-HERE.json",
    }
    (tmp_path / "EVAL-LOCK-CYCLE2.json").write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="chain is broken"):
        build_lock_state(tmp_path)


def test_the_generator_never_writes_a_lock_file(tmp_path: Path) -> None:
    """Prohibition, asserted: lock files are read-only to this package."""
    before = {p.name: p.read_bytes() for p in REPO_ROOT.glob("EVAL-LOCK*.json")}
    build_all(REPO_ROOT, out_dir=tmp_path)
    after = {p.name: p.read_bytes() for p in REPO_ROOT.glob("EVAL-LOCK*.json")}
    assert before == after
    assert not list(tmp_path.glob("EVAL-LOCK*.json"))


# ─────────────────────────────────────────────────────────────────────────────
# The ladder
# ─────────────────────────────────────────────────────────────────────────────


@has_results
def test_ladder_rows_carry_split_floors_and_the_oracle_gap() -> None:
    ladder = build_ladder(read_result_rows(RESULTS_DIR))
    assert ladder["rungs"]
    for row in ladder["rungs"]:
        assert row["split"] in SPLIT_VALUES
        for floor in (
            "savings_floor_random",
            "savings_floor_all_pass",
            "savings_floor_all_hold",
            "savings_floor_volume_rank",
        ):
            assert floor in row["metrics"], f"rung {row['label']} has no {floor}"
        assert "gap_to_oracle" in row["metrics"]
        assert "oracle_savings" in row["metrics"]
        assert row["provenance_consistent"], (
            f"{row['label']} aggregates rows from more than one harness: "
            f"{row['eval_lock_sha']} / {row['git_sha']}"
        )


@has_results
def test_no_ladder_row_is_split_free_or_prevalence_free() -> None:
    """FR-021: a PR-AUC without the prevalence it was measured at is the v1 mistake."""
    ladder = build_ladder(read_result_rows(RESULTS_DIR))
    for row in ladder["rungs"]:
        assert row["metrics"].get("prevalence") is not None
        assert row["split"] in SPLIT_VALUES


def test_ladder_refuses_an_empty_result_set() -> None:
    with pytest.raises(ArtifactSchemaError, match="do not "):
        build_ladder([])


def test_read_result_rows_refuses_a_row_missing_prevalence(tmp_path: Path) -> None:
    (tmp_path / "rung9_val_seed42.json").write_text(json.dumps({"rung": 9}), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="not an EvalResult"):
        read_result_rows(tmp_path)


def test_non_finite_metrics_become_null_with_the_reason_beside_them() -> None:
    """Constructed, so it does not depend on any current number."""
    base: dict[str, Any] = {
        f: 0.5 for f in ("prevalence", "pr_auc", "roc_auc", "ece", "savings", "gap_to_oracle")
    }
    row = {
        **base,
        "rung": 2,
        "split": "val",
        "label": "rung2",
        "ttd_median_days": float("inf"),
        "precision_at_k": float("nan"),
        "recall_by_typology": {"R1": float("nan")},
        "floor_fail": [],
        "eval_lock_sha": "a" * 64,
        "git_sha": "b" * 40,
        "open_count": 0,
        "_source": "rung2_val_seed42.json",
        "_seed": 42,
    }
    ladder = build_ladder([row, {**row, "_seed": 43, "_source": "rung2_val_seed43.json"}])
    entry = ladder["rungs"][0]
    assert entry["metrics"]["ttd_median_days"] is None
    assert entry["non_finite"]["ttd_median_days"] == {"Infinity": 2}
    assert entry["non_finite"]["precision_at_k"] == {"NaN": 2}
    assert entry["recall_by_typology"]["R1"] is None
    assert entry["non_finite"]["recall_by_typology"] == {"R1:NaN": 2}
    canonical_bytes(ladder)


# ─────────────────────────────────────────────────────────────────────────────
# G5
# ─────────────────────────────────────────────────────────────────────────────


def test_g5_builds_from_a_gate_dump_and_every_series_is_split_labelled() -> None:
    payload = build_g5(_g5_fixture())
    assert [s["detector"] for s in payload["series"]] == ["raw", "cohort-residual"]
    assert all(s["split"] == "NULL_RUN" for s in payload["series"])
    doc = envelope("g5_confounder_null", payload, split="NULL_RUN", provenance={})
    assert validate(doc) == "g5_confounder_null"


def test_g5_refuses_a_non_zero_prevalence_run() -> None:
    with pytest.raises(ArtifactSchemaError, match="zero prevalence"):
        build_g5({**_g5_fixture(), "prevalence": 0.015})


def test_g5_refuses_a_series_whose_length_disagrees_with_the_x_axis() -> None:
    bad = _g5_fixture()
    bad["series"][0]["alert_rate_by_day"] = [0.1, 0.2]
    with pytest.raises(ArtifactSchemaError, match="same length"):
        build_g5(bad)


def test_g5_missing_input_is_a_named_absence_not_a_fabricated_chart(tmp_path: Path) -> None:
    """The failure mode this whole ticket exists to prevent."""
    manifest = build_all(
        REPO_ROOT, g5_path=Path("data/v2/gates/does-not-exist.json"), out_dir=tmp_path
    )
    entry = next(a for a in manifest["artifacts"] if a["name"] == "g5_confounder_null")
    assert entry["status"] == "MISSING"
    assert entry["file"] is None and entry["sha256"] is None
    assert "does-not-exist.json" in entry["reason"]
    assert not (tmp_path / "g5_confounder_null.json").exists()


def test_g5_present_input_is_emitted_and_indexed(tmp_path: Path) -> None:
    dump = tmp_path / "g5_series.json"
    dump.write_text(json.dumps(_g5_fixture()), encoding="utf-8")
    out = tmp_path / "out"
    manifest = build_all(REPO_ROOT, g5_path=dump, out_dir=out)
    entry = next(a for a in manifest["artifacts"] if a["name"] == "g5_confounder_null")
    assert entry["status"] == "PRESENT"
    assert validate_file(out / "g5_confounder_null.json") == "g5_confounder_null"


# ─────────────────────────────────────────────────────────────────────────────
# End to end
# ─────────────────────────────────────────────────────────────────────────────


def test_every_emitted_file_validates_and_the_manifest_indexes_it(tmp_path: Path) -> None:
    manifest = build_all(REPO_ROOT, out_dir=tmp_path)
    on_disk = {p.name for p in tmp_path.glob("*.json")}
    assert "manifest.json" in on_disk
    for path in tmp_path.glob("*.json"):
        assert validate_file(path)
    indexed = {a["file"] for a in manifest["artifacts"] if a["status"] == "PRESENT"}
    assert indexed == on_disk - {"manifest.json"}


def test_manifest_sha256_matches_the_bytes_on_disk(tmp_path: Path) -> None:
    """The manifest's hashes are checkable by anyone with the repo and sha256sum."""
    import hashlib

    manifest = build_all(REPO_ROOT, out_dir=tmp_path)
    for entry in manifest["artifacts"]:
        if entry["status"] != "PRESENT":
            continue
        blob = (tmp_path / entry["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"]


def test_committed_artifacts_directory_is_in_contract(tmp_path: Path) -> None:
    """What is committed under ``artifacts/`` must itself validate, or the site is broken."""
    committed = REPO_ROOT / "artifacts"
    if not committed.is_dir():
        pytest.skip("artifacts/ has not been generated in this tree yet")
    for path in sorted(committed.glob("*.json")):
        assert validate_file(path)


# ─────────────────────────────────────────────────────────────────────────────
# The test split stays shut until T-0116 says otherwise
# ─────────────────────────────────────────────────────────────────────────────


def _row(split: str, **over: Any) -> dict[str, Any]:
    """A complete ``EvalResult`` row, every field filled from the dataclass itself.

    Derived rather than hand-listed so that a metric another lane adds upstream does not
    silently turn these tests into a check of ``read_result_rows``' field list instead of
    the thing each one is actually about.
    """
    row: dict[str, Any] = {
        field.name: 0.5 for field in fields(EvalResult) if field.name != "recall_by_typology"
    }
    row.update(
        rung=2,
        split=split,
        label="rung2",
        recall_by_typology={},
        cost_scenario="base",
        floor_fail=[],
        eval_lock_sha="a" * 64,
        git_sha="b" * 40,
        open_count=0,
    )
    row["_source"] = f"rung2_{split}_seed42.json"
    row["_seed"] = 42
    row.update(over)
    return row


def test_a_test_split_row_is_refused_while_the_open_counter_is_zero() -> None:
    """The hard constraint, as a test: no TEST number before T-0116 opens the split."""
    with pytest.raises(ArtifactSchemaError, match="open counter is 0"):
        build_ladder([_row("test")])


def test_a_test_split_row_is_refused_even_beside_valid_validation_rows() -> None:
    with pytest.raises(ArtifactSchemaError, match="rung2_test_seed42.json"):
        build_ladder([_row("val"), _row("test")])


def test_a_test_split_row_is_allowed_once_the_counter_says_it_was_opened() -> None:
    """After T-0116 the same row is legitimate — the guard is the counter, not a taboo."""
    ladder = build_ladder([_row("test")], test_split_opened=True)
    assert [r["split"] for r in ladder["rungs"]] == ["TEST"]


def test_no_emitted_artefact_contains_a_test_split_number(tmp_path: Path) -> None:
    """End to end over what this tree actually emits, not over a fixture."""
    build_all(REPO_ROOT, out_dir=tmp_path)
    for path in tmp_path.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert "TEST" not in json.dumps(doc), f"{path.name} carries a TEST-split label"


# ─────────────────────────────────────────────────────────────────────────────
# Provenance: the ladder may not borrow the live lock's authority
# ─────────────────────────────────────────────────────────────────────────────


@has_results
def test_ladder_provenance_reports_the_rows_own_shas_not_only_the_locks(tmp_path: Path) -> None:
    """A number computed at another commit must not read as one computed under this lock."""
    build_all(REPO_ROOT, out_dir=tmp_path)
    doc = json.loads((tmp_path / "ladder.json").read_text(encoding="utf-8"))
    prov = doc["provenance"]
    for key in ("results_eval_lock_sha", "results_git_sha", "harness_note"):
        assert key in prov, f"ladder provenance is missing {key!r}"
    assert prov["results_eval_lock_sha"] == [prov["eval_lock_sha"]]
    assert "geometry" in prov["harness_note"]


def test_rows_from_a_different_harness_are_a_refusal(tmp_path: Path) -> None:
    """The lock's whole point: results against another eval module are not comparable."""
    results = tmp_path / "eval"
    results.mkdir()
    row = _row("val", eval_lock_sha="f" * 64)
    del row["_source"], row["_seed"]
    (results / "rung2_val_seed42.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="authoritative lock"):
        build_all(REPO_ROOT, results_dir=results, out_dir=tmp_path / "out")


# ─────────────────────────────────────────────────────────────────────────────
# G5 windows carry their own role, so the figure is not hardcoded client-side
# ─────────────────────────────────────────────────────────────────────────────


def test_g5_windows_must_declare_adversarial_or_control() -> None:
    bad = _g5_fixture()
    del bad["windows"][0]["role"]
    with pytest.raises(ArtifactSchemaError, match="'role' is required"):
        build_g5(bad)


def test_g5_refuses_an_unknown_window_role() -> None:
    bad = _g5_fixture()
    bad["windows"][0]["role"] = "interesting"
    with pytest.raises(ArtifactSchemaError, match="role 'interesting'"):
        build_g5(bad)


def test_g5_refuses_a_window_that_falls_off_the_x_axis() -> None:
    bad = _g5_fixture(n_days=5)
    bad["windows"][0]["end_day"] = 99
    with pytest.raises(ArtifactSchemaError, match="off the x axis"):
        build_g5(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Drift: a mutated artefact must be caught, not rendered
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: d["payload"]["rungs"][0].pop("split"), id="row-loses-its-split"),
        pytest.param(lambda d: d.pop("provenance"), id="envelope-loses-provenance"),
        pytest.param(lambda d: d.update(schema_version="v3.0.1"), id="version-drifts"),
        pytest.param(
            lambda d: d["payload"]["rungs"][0].update(persona_id="M-0001"),
            id="ground-truth-leaks-in",
        ),
        pytest.param(lambda d: d["payload"].pop("metric_keys"), id="payload-key-disappears"),
    ],
)
@has_results
def test_a_drifted_artefact_on_disk_is_rejected_by_name(tmp_path: Path, mutate: Any) -> None:
    """What the ticket is for: the loader must never be handed a file that silently changed."""
    build_all(REPO_ROOT, out_dir=tmp_path)
    path = tmp_path / "ladder.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError) as exc:
        validate_file(path)
    assert exc.value.artifact in {"ladder", "ladder.json"}


@has_results
def test_the_committed_artifacts_are_what_this_generator_produces(tmp_path: Path) -> None:
    """Drift between the committed files and the emitter, caught here rather than in a demo.

    Skipped where ``data/`` is absent (it is gitignored, so CI has no results to rebuild
    from); locally it is the check that someone hand-edited an artefact.
    """
    committed = REPO_ROOT / "artifacts"
    if not committed.is_dir():
        pytest.skip("artifacts/ has not been generated in this tree yet")
    build_all(REPO_ROOT, out_dir=tmp_path)
    for path in sorted(tmp_path.glob("*.json")):
        assert (committed / path.name).exists(), f"{path.name} was never committed"
        assert (committed / path.name).read_bytes() == path.read_bytes(), (
            f"artifacts/{path.name} is not what `make artifacts` produces — either it was "
            "hand-edited or the generator changed without a rebuild"
        )
