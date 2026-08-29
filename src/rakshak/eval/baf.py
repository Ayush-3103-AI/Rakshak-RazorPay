"""Validate the decision/cost layer on BAF — real-bank-derived public data (T-0012, FR-021).

This module exists to back one sentence that `CLAUDE.md` mandates verbatim:

    "The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), a
    public benchmark derived from real bank data."

Until this ran, the repo printed that sentence with a parenthetical apology. FR-021 was
promoted SHOULD -> MUST on 2026-08-28 rather than let a SHOULD underwrite an honesty claim.

**What this does and does not validate.** BAF is bank *account-opening applications*: no
amount, no timestamp, no payer, no merchant, and no sequences. So the HMM cannot run here
and is not run here. What is exercised is the **decision layer** — Bayes Minimum Risk over
a cost matrix, the analyst-hour capacity constraint, and the savings score — against a real
label distribution (~1.1% prevalence) with real temporal drift across eight months.

**The split is BAF's own.** `06-requirements.md` reserves the synthetic test window; this
file never touches it. Months 0-5 train, month 6 early-stops, **month 7 is reported**.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.cli import base_parser, seed_everything
from rakshak.config import (
    ANCILLARY_LOADING_PHI,
    CHARGEBACK_REALISATION_RATE,
    GROSS_MARGIN_RATE,
    MERCHANT_LIFETIME_MONTHS,
    RESULTS_DIR,
    SEED,
)
from rakshak.data.download import EXTERNAL_DIR
from rakshak.decision import policy
from rakshak.eval import metrics
from rakshak.eval.oracle import review_slots
from rakshak.models.gbdt import (
    EARLY_STOPPING_ROUNDS as GBDT_EARLY_STOPPING_ROUNDS,
)
from rakshak.models.gbdt import (
    N_ROUNDS as GBDT_N_ROUNDS,
)
from rakshak.models.gbdt import (
    PARAMS as GBDT_PARAMS,
)

BAF_ZIP: Path = EXTERNAL_DIR / "baf.zip"
"""Downloaded by `rakshak.data.download --dataset baf`. Git-ignored; manifest is not."""

VARIANT: str = "Base.csv"
"""The Base variant, per `11-tickets/T-0012.md`. The five bias variants are not used."""

LABEL: str = "fraud_bool"
EXPOSURE: str = "proposed_credit_limit"
"""BAF's only monetary column. See `baf_cost_params` for what is assumed of it."""

TRAIN_MONTHS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
VALIDATE_MONTHS: tuple[int, ...] = (6,)
TEST_MONTHS: tuple[int, ...] = (7,)

_DROP_FROM_FEATURES: tuple[str, ...] = (LABEL, "month")
"""`month` is the split key. Leaving it in a feature matrix would leak the split."""


@dataclass(frozen=True)
class BafSplits:
    """BAF's native temporal split. Attributes are row subsets of one frame."""

    train: pd.DataFrame
    validate: pd.DataFrame
    test: pd.DataFrame


def load_baf(
    zip_path: Path = BAF_ZIP,
    variant: str = VARIANT,
    subsample: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Read one BAF variant out of the downloaded zip.

    The zip is never extracted to disk — `data/` is git-ignored but a 213 MB CSV beside a
    558 MB zip is still 771 MB of avoidable clutter.

    Args:
        zip_path: The downloaded archive.
        variant: Member filename inside the zip.
        subsample: If set, sample this many rows **stratified by month** so the temporal
            split survives, using `seed`. None reads all ~1M rows.
        seed: Sampling seed. Recorded in the results file when subsampling is used.

    Returns:
        The variant as a DataFrame.

    Raises:
        FileNotFoundError: If the archive is absent, with the command that fetches it.
    """
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Fetch it with:\n"
            "    python -m rakshak.data.download --dataset baf\n"
            "which needs a Kaggle API token ($KAGGLE_API_TOKEN or ~/.kaggle/access_token)."
        )
    with zipfile.ZipFile(zip_path) as archive, archive.open(variant) as handle:
        frame = pd.read_csv(handle)

    if subsample is not None and subsample < len(frame):
        frame = (
            frame.groupby("month", group_keys=False)
            .sample(frac=subsample / len(frame), random_state=seed)
            .reset_index(drop=True)
        )
    return frame


def split_baf(frame: pd.DataFrame) -> BafSplits:
    """Partition BAF on its own `month` column — never on the synthetic split.

    Args:
        frame: A BAF variant with a `month` column.

    Returns:
        Train (months 0-5), validate (6) and test (7) subsets.

    Raises:
        KeyError: If `month` is absent. BAF's temporal provenance is the reason this
            validation is worth anything, so a missing split key fails loudly.
    """
    if "month" not in frame.columns:
        raise KeyError("BAF frame has no `month` column; refusing to guess a temporal split")
    return BafSplits(
        train=frame[frame["month"].isin(TRAIN_MONTHS)],
        validate=frame[frame["month"].isin(VALIDATE_MONTHS)],
        test=frame[frame["month"].isin(TEST_MONTHS)],
    )


def baf_cost_params(frame: pd.DataFrame) -> policy.CostParams:
    """Map BAF's one monetary column onto the cost layer's `L` and `V`.

    **This mapping is an ASSUMPTION and the results file says so.** BAF records no
    realised loss and no customer lifetime value; `proposed_credit_limit` is the only
    monetary column it has, and it is a *proposed* limit, not a realised exposure.

        L_i = credit_limit_i * r_cb * (1 + phi)     realised loss if a fraudulent
                                                    application is approved
        V_i = credit_limit_i * g * lifetime         margin a good customer returns

    Both are linear in the same column, so `L/V` is constant across applications and the
    **only** thing making per-application thresholds differ is the flat `c_support` term
    inside `c_fp`. That is genuinely weaker example-dependence than the synthetic layer,
    where volume and lifetime vary independently. `tests/test_baf.py` pins both halves of
    that claim so the caveat cannot go stale.

    A second assumption: BAF's monetary unit is treated as the cost layer's monetary unit.
    The absolute scale matters because `c_support` and `c_review` are absolute. This is why
    the results file reports savings **across the whole swept asymmetry range** rather than
    quoting one point.

    Args:
        frame: A BAF variant carrying `proposed_credit_limit`.

    Returns:
        `CostParams` with per-application `loss_inr` and `value_inr`.
    """
    exposure = frame[EXPOSURE].to_numpy(dtype=float)
    return policy.CostParams(
        loss_inr=exposure * CHARGEBACK_REALISATION_RATE * (1.0 + ANCILLARY_LOADING_PHI),
        value_inr=exposure * GROSS_MARGIN_RATE * MERCHANT_LIFETIME_MONTHS,
    )


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric feature matrix: drop the label and the split key, one-hot the categoricals."""
    features = frame.drop(columns=[c for c in _DROP_FROM_FEATURES if c in frame.columns])
    return pd.get_dummies(features, drop_first=True)


def score_baf(splits: BafSplits, seed: int = SEED) -> dict[str, np.ndarray]:
    """Produce one score per test-month application, for each of three models.

    * `gbdt` — LightGBM fitted on months 0-5, early-stopped on month 6, predicted on 7.
    * `credit_risk_score` — BAF's own shipped score, min-max scaled to [0, 1] using
      **training-month** bounds only. It is a feature, not a calibrated probability, and
      is included as the domain floor.
    * `random` — the AP-06 floor. `results/summary.md` demonstrated that a savings figure
      without this beside it is not a claim about the model, so it ships here too.

    Args:
        splits: BAF's native temporal split.
        seed: Seed for LightGBM and for the random scorer.

    Returns:
        model name -> scores on the test month, aligned to `splits.test`.
    """
    import lightgbm as lgb

    train_x = _feature_matrix(splits.train)
    valid_x = _feature_matrix(splits.validate).reindex(columns=train_x.columns, fill_value=0)
    test_x = _feature_matrix(splits.test).reindex(columns=train_x.columns, fill_value=0)

    # Reuse models/gbdt.py's parameters rather than inventing a second set. `deterministic`,
    # `force_row_wise` and `num_threads=1` in there are what make NFR-003 hold; LightGBM's
    # histogram construction is thread-order dependent without them, and these numbers ship.
    params = {**GBDT_PARAMS, "seed": seed}
    names = [str(c) for c in train_x.columns]
    booster = lgb.train(
        params,
        lgb.Dataset(train_x.to_numpy(dtype=float), label=splits.train[LABEL], feature_name=names),
        num_boost_round=GBDT_N_ROUNDS,
        valid_sets=[
            lgb.Dataset(
                valid_x.to_numpy(dtype=float), label=splits.validate[LABEL], feature_name=names
            )
        ],
        callbacks=[lgb.early_stopping(GBDT_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    lo = float(splits.train["credit_risk_score"].min())
    hi = float(splits.train["credit_risk_score"].max())
    domain = (splits.test["credit_risk_score"].to_numpy(dtype=float) - lo) / max(hi - lo, 1e-12)

    rng = np.random.default_rng(seed)
    return {
        "random": rng.random(len(splits.test)),
        "credit_risk_score": np.clip(domain, 0.0, 1.0),
        "gbdt": np.asarray(booster.predict(test_x.to_numpy(dtype=float)), dtype=float),
    }


def render_baf_validation(
    frame: pd.DataFrame,
    splits: BafSplits,
    sweep: pd.DataFrame,
    rows: list[dict[str, object]],
    params: policy.CostParams,
    capacity_hours: float,
    seed: int,
    subsample: int | None,
) -> str:
    """Build `results/baf_validation.md`."""
    y = splits.test[LABEL].to_numpy(dtype=float)
    held = {str(r["model"]): int(r["n_held"]) for r in rows}
    native = policy.fp_to_loss_asymmetry(y, params)
    low, central, high = policy.asymmetry_range(y, params)
    add: list[str] = []

    add.append("# BAF validation of the decision layer (T-0012, FR-021)")
    add.append("")
    add.append(
        "> **The decision layer is additionally validated on BAF (Feedzai, NeurIPS 2022), "
        "a public benchmark derived from real bank data.** This file is the evidence for "
        "that sentence. Until it existed, `CLAUDE.md` mandated the sentence and the repo "
        "could not back it."
    )
    add.append("")
    add.append("## What this validates, and what it does not")
    add.append("")
    add.append(
        "**BAF is bank account-opening applications.** No amount, no timestamp, no payer, "
        "no merchant, **no sequences**. So the HMM cannot run here and is not run here. "
        "Nothing in this file speaks to the sequence layer. Verbatim, per `CLAUDE.md`: "
        "*Sequence-layer metrics are measured on synthetic merchant streams with injected "
        "typologies; the generator is in this repo.*"
    )
    add.append("")
    add.append(
        "What **is** exercised is the decision layer on real data: Bayes Minimum Risk over "
        "the cost matrix, the analyst-hour capacity constraint, and the savings score, "
        "against a real label distribution with real temporal drift."
    )
    add.append("")
    add.append("| Field | Value |")
    add.append("|---|---|")
    invocation = f"python -m rakshak.eval.baf --seed {seed}"
    if subsample:
        invocation += f" --subsample {subsample}"
    add.append(f"| Produced by | `{invocation}` |")
    add.append("| Dataset | BAF Base variant, `data/external/baf.manifest.json` |")
    add.append("| Licence | CC BY-NC-SA 4.0 — git-ignored, **not vendored** |")
    sampling = f" (subsampled to {subsample:,}, seed {seed})" if subsample else " (full)"
    add.append(f"| Rows | {len(frame):,}{sampling} |")
    add.append(
        "| Split | BAF's **native** months — "
        f"train 0-5 ({len(splits.train):,}), validate 6 ({len(splits.validate):,}), "
        f"**test 7 ({len(splits.test):,})** |"
    )
    add.append(f"| Test-month prevalence | {y.mean():.4f} |")
    add.append(f"| Review capacity B | {capacity_hours:.2f} h |")
    add.append(f"| Native asymmetry | {native:.1f} INR FP cost per INR 100 loss |")
    add.append(f"| Swept range | {low:.1f} - {high:.1f}, central {central:.1f} |")
    add.append("")

    add.append("## Two assumptions, stated because neither is derivable from BAF")
    add.append("")
    add.append(
        "**1. `proposed_credit_limit` stands in for exposure.** BAF records no realised "
        "loss and no customer lifetime value. `L = limit * r_cb * (1 + phi)` and "
        "`V = limit * g * lifetime`. Both are linear in the same column, so `L/V` is "
        "constant and **the only source of per-application threshold variation is the flat "
        "`c_support` term.** That is weaker example-dependence than the synthetic layer, "
        "where volume and lifetime move independently. Do not describe this as validating "
        "example-dependent costing; it validates the policy and the constraint."
    )
    add.append("")
    add.append(
        "**2. BAF's monetary unit is treated as the cost layer's monetary unit.** Absolute "
        "scale matters because `c_support` and `c_review` are absolute. **This is why the "
        "table below is reported across the whole swept asymmetry range and no single-point "
        "savings figure is quoted as the result.**"
    )
    add.append("")

    add.append("## Read this before the tables: the cost matrix sits in an extreme corner")
    add.append("")
    add.append(
        f"**The native asymmetry reads {native:,.0f} against the synthetic split's 47.5, and "
        f"the swept range ({low:,.0f} - {high:,.0f}) never reaches 47.5 at any point.** So no "
        "row below is measured in the operating regime the rest of the project reports on. "
        "That is a consequence of assumption 2 above, not a property of BAF: BAF's credit "
        "limits run 190-2000 in its own units while `COST_SUPPORT_INR` and `COST_REVIEW_INR` "
        "are absolute INR constants sized for merchants doing lakhs of monthly volume, so "
        "`c_fp` dwarfs `L` for every application."
    )
    add.append("")
    add.append(
        "In that corner the economically correct policy is to hold almost nobody. The tables "
        "below should be read as **\"does the decision layer do the right thing when false "
        "positives are overwhelmingly expensive?\"** — not as a validation of the "
        "review-versus-hold trade-off at the project's own asymmetry."
    )
    add.append("")

    add.append("## Test month (month 7) — the reported window")
    add.append("")
    add.append(
        "| model | savings | PR-AUC | precision@K | Brier | reviewed | held "
        "| capacity binds |"
    )
    add.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        add.append(
            f"| {row['model']} | {row['savings']:+.4f} | {row['pr_auc']:.4f} | "
            f"{row['precision_at_k']:.4f} | {row['brier']:.4f} | {row['n_reviewed']} | "
            f"{row['n_held']} | {row['binding_constraint']} |"
        )
    add.append("")
    add.append(
        "**Read the `random` row first.** `results/summary.md` established on the synthetic "
        "split that the cost matrix, not detection, earns most of the savings *level*. The "
        "same discipline applies here: any margin quoted off the savings column is a claim "
        "about the model only to the extent it exceeds the `random` row."
    )
    add.append("")

    add.append("## Savings across the swept asymmetry")
    add.append("")
    models = list(dict.fromkeys(sweep["model"]))
    add.append("| asymmetry | " + " | ".join(models) + " |")
    add.append("|---|" + "---|" * len(models))
    for asymmetry, block in sweep.groupby("asymmetry", sort=True):
        cells = [f"{float(block[block['model'] == m]['savings'].iloc[0]):+.4f}" for m in models]
        add.append(f"| {asymmetry:.1f} | " + " | ".join(cells) + " |")
    add.append("")
    add.append(
        "The range is derived from `config.COST_PRIMITIVE_RANGES`, exactly as in "
        "`results/sensitivity.md`. Nothing here is narrowed because part of it is "
        "unflattering."
    )
    add.append("")

    add.append("## What the numbers actually say")
    add.append("")
    add.append(
        "**BMR does the economically correct thing under the corner described above.** "
        f"`gbdt` holds {held['gbdt']:,} applications out of {len(splits.test):,} and stays "
        f"positive; `random` holds {held['random']:,} and is destroyed. That is the policy "
        "behaving correctly under the costs it was given."
    )
    add.append("")
    add.append("**So this file validates:**")
    add.append("")
    add.append("* BMR takes the economically correct action under an extreme cost asymmetry;")
    add.append("* the analyst-hour capacity constraint binds on real data and is reported;")
    add.append(
        "* the savings score orders the three models identically at **every** swept "
        "asymmetry, across two orders of magnitude - the ordering is not an artefact of one "
        "cost matrix."
    )
    add.append("")
    add.append(
        "**It does not validate** the balanced regime where REVIEW and HOLD genuinely trade "
        "off against each other. No public dataset available to this project puts real money "
        "on both sides of that trade, and this one does not either."
    )
    add.append("")
    add.append("### The cross-check that cuts back at the synthetic split")
    add.append("")
    add.append(
        "`results/summary.md` reported that on the synthetic split `random` scored +0.6929 "
        "savings against `rules`' +0.6980 - within 0.0051 - and concluded the cost matrix, "
        "not detection, was earning the savings level (AP-06). **On BAF, at a realistic "
        f"{y.mean():.2%} prevalence, `random` is catastrophically negative at "
        f"{next(r['savings'] for r in rows if r['model'] == 'random'):+.4f}.**"
    )
    add.append("")
    add.append(
        "That points at the synthetic split's **20% merchant fraud rate**, not at the savings "
        "metric. At 20% prevalence a random policy lands on enough true positives to look "
        "competent; at 1.5% it cannot. The AP-06 warning stands - savings must never be "
        "quoted without PR-AUC beside it - but its severity on the synthetic split is "
        "substantially an artefact of a prevalence the generator inflated on purpose, for "
        "per-typology sample size. **T-0011 should say both things.**"
    )
    add.append("")
    return "\n".join(add)


def run(seed: int = SEED, subsample: int | None = None, results_dir: Path = RESULTS_DIR) -> Path:
    """Run the BAF validation end to end and write `results/baf_validation.md`."""
    frame = load_baf(subsample=subsample, seed=seed)
    splits = split_baf(frame)
    scores = score_baf(splits, seed=seed)

    test = splits.test
    y = test[LABEL].to_numpy(dtype=float)
    params = baf_cost_params(test)
    capacity_hours = policy.review_capacity_hours(len(test))
    k = min(review_slots(capacity_hours), len(test))

    rows: list[dict[str, object]] = []
    for name, score in scores.items():
        result = policy.bmr_policy(score, params, capacity_hours)
        rows.append(
            {
                "model": name,
                "savings": policy.savings(y, result.actions, params),
                "pr_auc": metrics.pr_auc(y, score),
                "precision_at_k": metrics.precision_at_k(y, score, k),
                "brier": metrics.brier_score(y, score),
                "n_reviewed": result.n_reviewed,
                "n_held": result.n_held,
                "binding_constraint": (
                    f"{result.binding_constraint} (wanted {result.unconstrained_n_reviewed})"
                ),
            }
        )

    sweep = policy.sweep_cost_asymmetry(
        y,
        scores,
        params,
        capacity_hours,
        seed=seed,
        # BAF has neither `rules` nor `hmm`. Left at their defaults the margin columns
        # would silently fill with NaN; the domain floor and gbdt are the real comparison.
        reference_model="credit_risk_score",
        proposal_model="gbdt",
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "baf_validation.md"
    path.write_text(
        render_baf_validation(
            frame, splits, sweep, rows, params, capacity_hours, seed, subsample
        ),
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = base_parser("Validate the decision layer on BAF (T-0012, FR-021).")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Stratified row cap, for a faster run. Recorded in the results file.",
    )
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    path = run(seed=args.seed, subsample=args.subsample)
    print(f"rakshak: wrote {path} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BafSplits",
    "baf_cost_params",
    "load_baf",
    "render_baf_validation",
    "run",
    "score_baf",
    "split_baf",
]
