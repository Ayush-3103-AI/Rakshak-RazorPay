"""The typer CLI. Every ``make`` target calls through here rather than into a module.

One entry point per pipeline stage, so that the Makefile stays a list of names and the
argument handling lives in exactly one place. Other lanes add their own subcommands
(``features``, ``train``, ``eval``, ``report``); this file owns ``gen``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import typer

from rakshak.generator.config import ScenarioConfig, load_scenario
from rakshak.generator.engine import generate

app = typer.Typer(add_completion=False, help="Rakshak v2 — merchant risk sentinel.")


@app.callback()
def main() -> None:
    """Rakshak v2. Run ``rakshak <command> --help`` for a stage."""


@app.command()
def gen(
    # B008 is suppressed on the two Path defaults only: `typer.Option(Path(...))` is
    # typer's documented way to give an option a path default, and the inner Path() is
    # what ruff sees. Rewriting it as a module-level singleton would obscure the CLI
    # signature to satisfy a rule aimed at mutable defaults, which Path is not.
    config: Path = typer.Option(  # noqa: B008
        Path("configs/scenario_v2.yaml"), "--config", "-c", help="Scenario manifest."
    ),
    seed: int = typer.Option(42, "--seed", help="Overrides the seed in the manifest."),
    out: Path = typer.Option(  # noqa: B008
        Path("data/v2"), "--out", "-o", help="Output directory."
    ),
    merchants: int | None = typer.Option(
        None, "--merchants", help="Override population.n_merchants (smoke runs, gates)."
    ),
    days: int | None = typer.Option(None, "--days", help="Override population.n_days."),
    prevalence: float | None = typer.Option(
        None,
        "--prevalence",
        help="Override population.prevalence. 0.0 is the gate-G5 confounder-null run.",
    ),
    confounders: bool = typer.Option(
        True, "--confounders/--no-confounders", help="Toggle the P1-P6 platform layer."
    ),
) -> None:
    """Generate the v2 dataset from a scenario manifest.

    Deterministic in ``--seed``: two runs at the same seed produce byte-identical tables,
    and gate G3 asserts exactly that.
    """
    scenario = _apply_overrides(
        load_scenario(config),
        merchants=merchants,
        days=days,
        prevalence=prevalence,
        confounders=confounders,
    )
    data = generate(scenario, np.random.default_rng(seed))
    paths = data.write(out)

    summary = {
        "seed": seed,
        "config": str(config),
        "n_merchants": scenario.population.n_merchants,
        "n_days": scenario.population.n_days,
        "prevalence": scenario.population.prevalence,
        "confounders_enabled": scenario.confounders.enabled,
        "analyst_capacity_k": scenario.analyst_capacity,
        "rows": {name: getattr(data, name).height for name in paths},
        "content_sha256": data.sha256(),
    }
    (Path(out) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


def _apply_overrides(
    scenario: ScenarioConfig,
    *,
    merchants: int | None,
    days: int | None,
    prevalence: float | None,
    confounders: bool,
) -> ScenarioConfig:
    """CLI overrides, applied to the loaded manifest.

    Overrides exist for the gates and for smoke runs, not as a second configuration
    surface: ``run_summary.json`` records every one of them next to the content hash, so
    a dataset can never be mistaken for the manifest's default population.
    """
    population = scenario.population
    if merchants is not None:
        population = dataclasses.replace(population, n_merchants=merchants)
    if days is not None:
        population = dataclasses.replace(population, n_days=days)
    if prevalence is not None:
        population = dataclasses.replace(population, prevalence=prevalence)
    return dataclasses.replace(
        scenario,
        population=population,
        confounders=dataclasses.replace(scenario.confounders, enabled=confounders),
    )


if __name__ == "__main__":  # pragma: no cover - `python -m rakshak.cli`
    app()
