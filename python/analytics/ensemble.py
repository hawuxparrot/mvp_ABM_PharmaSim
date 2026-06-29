"""Monte Carlo ensembles + sensitivity sweeps over the fraud-experiment loop.

A single run of :func:`analytics.experiment.run_fraud_experiment` reports one
precision/recall/F1 triple, but the engine is stochastic (Poisson demand, randomized
movement) and the perturbation picks targets at random, so a single triple is noise. This
module runs the loop over many seeds and reports metric *distributions* (mean + normal-
approx confidence interval) instead, and adds a sensitivity sweep over the detector
threshold that reuses each seed's simulated event log (the expensive part) so only the
cheap detection step repeats per threshold.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from analytics.fraud import DetectionMetrics, evaluate_detection
from compiler.types import EngineInput
from policy.perturbations import PerturbationLabel, PerturbationSpec, apply_perturbation


@dataclass
class MetricStat:
    """Mean, sample std, and a normal-approx CI for one scalar metric."""

    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass
class MetricSummary:
    n: int
    precision: MetricStat
    recall: MetricStat
    f1: MetricStat


@dataclass
class RunArtifacts:
    """Everything needed to (re)score a single seeded run without re-simulating."""

    seed: int
    spec: PerturbationSpec
    baseline_records: list[dict[str, Any]]
    perturbed_records: list[dict[str, Any]]
    perturbed_input: EngineInput
    labels: list[PerturbationLabel]


@dataclass
class EnsembleResult:
    seeds: list[int]
    runs: list[DetectionMetrics]
    summary: MetricSummary


@dataclass
class SweepPoint:
    value: float
    summary: MetricSummary


def _stat(values: Sequence[float], *, z: float) -> MetricStat:
    n = len(values)
    if n == 0:
        return MetricStat(0.0, 0.0, 0.0, 0.0)
    mean = sum(values) / n
    if n < 2:
        return MetricStat(mean, 0.0, mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    sem = std / math.sqrt(n)
    return MetricStat(mean, std, mean - z * sem, mean + z * sem)


def summarize_metrics(runs: Sequence[DetectionMetrics], *, z: float = 1.96) -> MetricSummary:
    """Aggregate per-run metrics into mean/std/CI. ``z=1.96`` is the 95% normal quantile."""
    return MetricSummary(
        n=len(runs),
        precision=_stat([r.precision for r in runs], z=z),
        recall=_stat([r.recall for r in runs], z=z),
        f1=_stat([r.f1 for r in runs], z=z),
    )


def simulate_runs(
    base_input: EngineInput,
    spec: PerturbationSpec,
    *,
    seeds: Sequence[int],
    ticks: int = 300,
    vary_spec_seed: bool = True,
) -> list[RunArtifacts]:
    """Run the baseline+perturbed pair once per seed and cache their event logs.

    Each seed reseeds the engine RNG (``EngineInput.seed``); with ``vary_spec_seed`` the
    perturbation's target selection is reseeded too, so both sources of stochasticity vary.
    Baseline and perturbed share a seed within a run, so the only systematic difference is
    the injected fraud.
    """
    # Lazy import: runtime.experiments -> canonical -> analytics.fraud -> analytics package,
    # so importing it at module top would form a cycle when analytics/__init__ loads ensemble.
    from runtime.experiments import run_simulation

    artifacts: list[RunArtifacts] = []
    for s in seeds:
        run_spec = replace(spec, seed=int(s)) if vary_spec_seed else spec
        inp = copy.deepcopy(base_input)
        inp.seed = int(s)
        _, baseline_records = run_simulation(inp, ticks)

        perturbed_input, labels = apply_perturbation(inp, run_spec)
        _, perturbed_records = run_simulation(perturbed_input, ticks)

        artifacts.append(
            RunArtifacts(
                seed=int(s),
                spec=run_spec,
                baseline_records=baseline_records,
                perturbed_records=perturbed_records,
                perturbed_input=perturbed_input,
                labels=labels,
            )
        )
    return artifacts


def score_runs(
    artifacts: Sequence[RunArtifacts], *, z_threshold: float
) -> list[DetectionMetrics]:
    """Detect + score each cached run at a given detector threshold (no re-simulation)."""
    from analytics.experiment import _detect_for

    out: list[DetectionMetrics] = []
    for a in artifacts:
        predicted = _detect_for(
            a.spec,
            a.perturbed_records,
            a.baseline_records,
            a.perturbed_input,
            z_threshold=z_threshold,
        )
        truth_ids = {l.location_ext_id for l in a.labels}
        out.append(evaluate_detection(set(predicted.keys()), truth_ids))
    return out


def run_fraud_ensemble(
    base_input: EngineInput,
    spec: PerturbationSpec,
    *,
    seeds: Sequence[int],
    ticks: int = 300,
    z_threshold: float = 3.0,
    vary_spec_seed: bool = True,
    z: float = 1.96,
) -> EnsembleResult:
    """Monte Carlo the fraud loop over ``seeds`` and report metric mean/CI."""
    artifacts = simulate_runs(
        base_input, spec, seeds=seeds, ticks=ticks, vary_spec_seed=vary_spec_seed
    )
    runs = score_runs(artifacts, z_threshold=z_threshold)
    return EnsembleResult(
        seeds=[a.seed for a in artifacts],
        runs=runs,
        summary=summarize_metrics(runs, z=z),
    )


def sweep_detection_threshold(
    base_input: EngineInput,
    spec: PerturbationSpec,
    *,
    thresholds: Sequence[float],
    seeds: Sequence[int],
    ticks: int = 300,
    vary_spec_seed: bool = True,
    z: float = 1.96,
) -> list[SweepPoint]:
    """Sweep the detector's ``z_threshold`` reusing one simulated ensemble per seed.

    Simulation dominates cost, so we simulate each seed once and re-run only the (cheap)
    detection+scoring step at each threshold -- a clean sensitivity curve for the
    precision/recall trade-off of the volume-change detectors.
    """
    artifacts = simulate_runs(
        base_input, spec, seeds=seeds, ticks=ticks, vary_spec_seed=vary_spec_seed
    )
    points: list[SweepPoint] = []
    for t in thresholds:
        runs = score_runs(artifacts, z_threshold=float(t))
        points.append(SweepPoint(value=float(t), summary=summarize_metrics(runs, z=z)))
    return points
