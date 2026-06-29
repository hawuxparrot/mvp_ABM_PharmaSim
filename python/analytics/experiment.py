"""One-call fraud experiment: baseline run -> perturb -> perturbed run -> detect -> score.

This closes the loop the project set out to demonstrate: a perturbation rewrites engine
parameters, the native engine produces an observable ground-truth log, a detector that only
sees that log flags suspect locations, and we score it against the injected labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analytics.fraud import (
    DetectionMetrics,
    detect_cross_market_moves,
    detect_theft,
    detect_volume_change,
    evaluate_detection,
)
from compiler.types import EngineInput
from policy.perturbations import (
    PerturbationLabel,
    PerturbationSpec,
    PerturbationType,
    apply_perturbation,
)
from runtime.experiments import build_bulgaria_simulation_input, run_simulation


@dataclass
class FraudExperimentResult:
    spec: PerturbationSpec
    baseline_input: EngineInput
    perturbed_input: EngineInput
    baseline_records: list[dict[str, Any]]
    perturbed_records: list[dict[str, Any]]
    labels: list[PerturbationLabel]
    predicted: dict[str, float]
    metrics: DetectionMetrics


def _detect_for(
    spec: PerturbationSpec,
    perturbed_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    inp: EngineInput,
    *,
    z_threshold: float,
) -> dict[str, float]:
    if spec.ptype in (PerturbationType.VOLUME_SPIKE, PerturbationType.COUNTERFEIT):
        return detect_volume_change(
            perturbed_records, baseline_records, inp, z_threshold=z_threshold
        )
    if spec.ptype == PerturbationType.SMUGGLING:
        return detect_cross_market_moves(perturbed_records, inp)
    if spec.ptype == PerturbationType.THEFT:
        return detect_theft(perturbed_records, inp)
    raise ValueError(f"no detector mapped for {spec.ptype!r}")


def run_fraud_experiment(
    spec: PerturbationSpec,
    *,
    bg_registry_path: str | Path = "data/data.json",
    spor_locations_path: str | Path = "data/spor_locations.csv",
    geocode: bool = False,
    geocode_cache_path: str | Path | None = None,
    max_locations: int = 350,
    packs_per_obp: int = 60,
    packs_per_wholesaler: int = 12,
    n_obp_hubs: int = 6,
    ticks: int = 300,
    z_threshold: float = 3.0,
) -> FraudExperimentResult:
    si = build_bulgaria_simulation_input(
        bg_registry_path=bg_registry_path,
        spor_locations_path=spor_locations_path,
        geocode=geocode,
        geocode_cache_path=geocode_cache_path,
        max_locations=max_locations,
        packs_per_obp=packs_per_obp,
        packs_per_wholesaler=packs_per_wholesaler,
        n_obp_hubs=n_obp_hubs,
    )
    baseline_input = si.engine_input
    _, baseline_records = run_simulation(baseline_input, ticks)

    perturbed_input, labels = apply_perturbation(baseline_input, spec)
    _, perturbed_records = run_simulation(perturbed_input, ticks)

    predicted = _detect_for(
        spec,
        perturbed_records,
        baseline_records,
        perturbed_input,
        z_threshold=z_threshold,
    )
    truth_ids = {l.location_ext_id for l in labels}
    metrics = evaluate_detection(set(predicted.keys()), truth_ids)

    return FraudExperimentResult(
        spec=spec,
        baseline_input=baseline_input,
        perturbed_input=perturbed_input,
        baseline_records=baseline_records,
        perturbed_records=perturbed_records,
        labels=labels,
        predicted=predicted,
        metrics=metrics,
    )
