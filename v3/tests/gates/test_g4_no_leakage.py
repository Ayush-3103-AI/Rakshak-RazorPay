"""G4 — no leakage. Ground-truth fields are unreachable from features and models.

**Blocking.** Prime Directive 3: ``persona_id``, ``risk_typology_id``, ``drift_onset_at``,
``true_loss_amount_inr``, ``is_unreported`` and ``GroundTruth`` itself must never be
reachable from ``src/rakshak/features/`` or ``src/rakshak/models/``. Leakage invalidates
every number in the project, and it does so silently and flatteringly, which is the worst
combination available.

The scan is AST-based rather than textual. A substring search would both miss
``frame["drift_onset_at"]`` reached through a variable and fire on the word appearing in
a docstring that explains the quarantine. Imports, attribute access, bare names and
string literals are all checked; docstrings are excluded.

G4 has a second clause in the spec - point-in-time recomputation at time t must match
the stored feature vector exactly. That needed a feature layer, which did not exist
when this gate was first written; Lane B has since landed one, so G4b below now runs
against real generator output rather than recording a SKIP.

The distinction matters. ``tests/parity/`` proves the two runners agree on a SYNTHETIC
stream, and T-120 found a bug that survived exactly that check: warmup was anchored on
``onboarded_at`` while the stream starts at day 0, so every baseline was empty and both
runners agreed perfectly on the wrong answer. Recomputing against real generator output
is the half that catches a feature which is servable and wrong.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import polars as pl
from gates_report import GATE_SEED, green_if
from parity_harness import ParityFailure, assert_parity

from rakshak.features import registry, tier1
from rakshak.generator.engine import GeneratedData
from rakshak.schemas import (
    RADIOACTIVE_FIELDS,
    Instrument,
    MerchantProfile,
    Transaction,
    TxnStatus,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "rakshak"
QUARANTINED_FROM = ("features", "models")


def _string_constants(tree: ast.AST) -> set[str]:
    """String literals, minus docstrings.

    A radioactive field reaches a model just as well through ``frame["drift_onset_at"]``
    as through an import, and that form has no name node to catch.
    """
    docstrings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.extend(
                f"{path.name}:{node.lineno} imports {alias.name}"
                for alias in node.names
                if alias.name in RADIOACTIVE_FIELDS
            )
        elif isinstance(node, ast.Attribute) and node.attr in RADIOACTIVE_FIELDS:
            found.append(f"{path.name}:{node.lineno} attribute .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in RADIOACTIVE_FIELDS:
            found.append(f"{path.name}:{node.lineno} name {node.id}")
    found.extend(
        f"{path.name} string literal {literal!r}"
        for literal in sorted(_string_constants(tree) & RADIOACTIVE_FIELDS)
    )
    return found


def test_g4_no_ground_truth_reaches_features_or_models() -> None:
    offenders: list[str] = []
    scanned = 0
    for package in QUARANTINED_FROM:
        for path in sorted((SRC / package).rglob("*.py")):
            scanned += 1
            offenders.extend(f"{package}/{item}" for item in scan(path))
    ok = green_if(
        "G4 no-leakage",
        not offenders,
        f"{len(offenders)} forbidden reference(s) across {scanned} file(s) in "
        f"{'/'.join(QUARANTINED_FROM)}",
        "; ".join(offenders) if offenders else f"quarantined: {sorted(RADIOACTIVE_FIELDS)}",
    )
    assert ok, f"ground-truth leakage: {offenders}"


def test_g4_the_scanner_actually_catches_leakage(tmp_path: Path) -> None:
    """A clean scan of a nearly empty package is not evidence that the scanner works.

    Lane B's files land later. This proves the gate will see them when they do — and it
    is the reason a green G4 today means anything at all.
    """
    leaky = (
        "from rakshak.schemas import GroundTruth\n",
        "def f(gt):\n    return gt.drift_onset_at\n",
        "def f(frame):\n    return frame['risk_typology_id']\n",
        "def f(row):\n    return row[chr(0x70) + 'ersona_id']\n",
    )
    probe = tmp_path / "probe.py"
    for i, snippet in enumerate(leaky[:3]):
        probe.write_text(snippet, encoding="utf-8")
        assert scan(probe), f"scanner missed leak #{i}: {snippet!r}"

    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""This docstring mentions drift_onset_at only in prose."""\nX = 1\n', encoding="utf-8"
    )
    assert not scan(clean)


def test_g4_point_in_time_recomputation_matches_the_online_state(
    gate_data: GeneratedData,
) -> None:
    """G4's second clause, against real generator output.

    For a sample of merchants, fold the real event stream through every registered
    feature's ``update()`` and compare, at each epoch, against ``batch()`` recomputed from
    the point-in-time prefix. A disagreement means the value a model would be trained on is
    not the value the online path would have produced — the same number by two routes, and
    only one of those routes is the one that would actually run.

    Sampled rather than exhaustive, for two reasons. 14.8M transactions will not become
    Python objects on a laptop; and a leak of this kind is a property of the feature, not
    of the merchant, so a sample that covers every feature is the relevant coverage. The
    sample is seeded, and it deliberately over-weights the fraud population: a feature that
    is only wrong on the drifting merchants is the one that would flatter every metric in
    the project while looking correct everywhere a casual check would land.
    """
    profiles_by_id = {
        row["merchant_id"]: MerchantProfile(
            merchant_id=row["merchant_id"],
            onboarded_at=row["onboarded_at"],
            mcc=row["mcc"],
            mcc_group=row["mcc_group"],
            declared_monthly_gmv=row["declared_monthly_gmv"],
            kyc_tier=row["kyc_tier"],
            vintage_months=row["vintage_months"],
            city_tier=row["city_tier"],
        )
        for row in gate_data.profiles.iter_rows(named=True)
    }

    # ground_truth is readable here: tests/gates/ is not features/ or models/, and the
    # quarantine G4a enforces is about what the SYSTEM can reach, not what a gate can.
    fraud = set(
        gate_data.ground_truth.filter(pl.col("risk_typology_id").is_not_null())[
            "merchant_id"
        ].to_list()
    )
    clean = sorted(set(profiles_by_id) - fraud)
    rng = np.random.default_rng(GATE_SEED)
    sample = sorted(
        {
            *rng.choice(sorted(fraud), size=min(12, len(fraud)), replace=False).tolist(),
            *rng.choice(clean, size=min(24, len(clean)), replace=False).tolist(),
        }
    )

    frame = gate_data.transactions.filter(pl.col("merchant_id").is_in(sample))
    txns = [
        Transaction(
            event_id=r["event_id"],
            merchant_id=r["merchant_id"],
            payer_id=r["payer_id"],
            event_time=r["event_time"],
            event_date=r["event_date"],
            amount_inr=r["amount_inr"],
            instrument=Instrument(r["instrument"]),
            is_cnp=r["is_cnp"],
            is_international=r["is_international"],
            bin_hash=r["bin_hash"],
            device_hash=r["device_hash"],
            ip_hash=r["ip_hash"],
            status=TxnStatus(r["status"]),
            decline_code=r["decline_code"],
            mcc=r["mcc"],
            is_refund=r["is_refund"],
            refund_of=r["refund_of"],
        )
        for r in frame.iter_rows(named=True)
    ]
    sampled_profiles = {m: profiles_by_id[m] for m in sample}
    tier1.load_profiles(sampled_profiles)

    failure = ""
    for spec in registry.REGISTRY.values():
        try:
            assert_parity(spec, txns, sampled_profiles)
        except ParityFailure as exc:
            failure = f"{spec.name}: {exc}"
            break

    green_if(
        "G4b point-in-time",
        not failure,
        (
            f"{len(registry.REGISTRY)} features x {len(sample)} merchants "
            f"({len(txns):,} real events) agree online vs offline at every epoch"
        )
        if not failure
        else f"DISAGREEMENT {failure}",
        "recomputed from real generator output, not a synthetic stream - T-120 found a "
        "warmup bug that survived the synthetic check because both runners were wrong",
    )
    assert not failure, failure
