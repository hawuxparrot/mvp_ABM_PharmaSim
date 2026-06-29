"""Fraud/smuggling perturbation loop: perturb engine params -> simulate -> detect -> score."""

from __future__ import annotations

import importlib.util

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from analytics.experiment import run_fraud_experiment  # noqa: E402
from policy.perturbations import (  # noqa: E402
    PerturbationSpec,
    PerturbationType,
    apply_perturbation,
)
from runtime.experiments import build_bulgaria_simulation_input  # noqa: E402

EXP_KW = dict(max_locations=300, packs_per_obp=40, packs_per_wholesaler=10, ticks=250)


def test_apply_perturbation_does_not_mutate_baseline() -> None:
    si = build_bulgaria_simulation_input(max_locations=200)
    base = si.engine_input
    n_packs_before = base.n_packs
    markets_before = list(base.market_code)
    spec = PerturbationSpec(ptype=PerturbationType.SMUGGLING, n_targets=2)
    perturbed, labels = apply_perturbation(base, spec)
    assert base.n_packs == n_packs_before
    assert base.market_code == markets_before
    assert perturbed is not base
    assert labels


@pytest.mark.parametrize(
    "ptype",
    [
        PerturbationType.VOLUME_SPIKE,
        PerturbationType.SMUGGLING,
        PerturbationType.THEFT,
        PerturbationType.COUNTERFEIT,
    ],
)
def test_fraud_experiment_detects_injected_anomaly(ptype: PerturbationType) -> None:
    spec = PerturbationSpec(ptype=ptype, n_targets=3, intensity=8.0, seed=7)
    res = run_fraud_experiment(spec, **EXP_KW)
    assert res.labels
    # Baseline detector must recover every injected location (recall) with no false alarms.
    assert res.metrics.recall == pytest.approx(1.0)
    assert res.metrics.precision == pytest.approx(1.0)


def test_smuggling_changes_pack_markets() -> None:
    spec = PerturbationSpec(ptype=PerturbationType.SMUGGLING, n_targets=3, seed=7)
    res = run_fraud_experiment(spec, **EXP_KW)
    inp = res.perturbed_input
    cross = 0
    for r in res.perturbed_records:
        if r["event_type"] != "MOVE":
            continue
        to_id = r["to_location_id"]
        if to_id is None:
            continue
        if int(inp.pack_initial_market_id[r["pack_id"]]) != int(inp.location_market_id[to_id]):
            cross += 1
    assert cross > 0


def test_theft_creates_supply_node_decommissions() -> None:
    spec = PerturbationSpec(ptype=PerturbationType.THEFT, n_targets=3, seed=7)
    res = run_fraud_experiment(spec, **EXP_KW)
    # Theft detector keys off decommissions at supply nodes, which baseline lacks.
    assert res.predicted
