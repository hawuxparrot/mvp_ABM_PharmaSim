"""Phase 6: Monte Carlo ensembles + detector-threshold sensitivity sweep."""

from __future__ import annotations

import importlib.util

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from analytics.ensemble import (  # noqa: E402
    run_fraud_ensemble,
    summarize_metrics,
    sweep_detection_threshold,
)
from analytics.fraud import DetectionMetrics  # noqa: E402
from policy.perturbations import PerturbationSpec, PerturbationType  # noqa: E402
from runtime.experiments import build_bulgaria_simulation_input  # noqa: E402

BUILD_KW = dict(max_locations=200, packs_per_obp=40, packs_per_wholesaler=10)


def _metrics(p: float, r: float, f1: float) -> DetectionMetrics:
    return DetectionMetrics(0, 0, 0, p, r, f1)


def test_summarize_metrics_mean_and_ci() -> None:
    runs = [_metrics(1.0, 1.0, 1.0), _metrics(0.0, 0.0, 0.0)]
    s = summarize_metrics(runs)
    assert s.n == 2
    assert s.precision.mean == pytest.approx(0.5)
    # CI brackets the mean and std is the sample std of {0,1}.
    assert s.precision.ci_low < s.precision.mean < s.precision.ci_high
    assert s.precision.std == pytest.approx(0.7071, abs=1e-3)


def test_summarize_metrics_single_run_zero_width_ci() -> None:
    s = summarize_metrics([_metrics(0.8, 0.6, 0.7)])
    assert s.n == 1
    assert s.f1.ci_low == s.f1.ci_high == pytest.approx(0.7)
    assert s.f1.std == 0.0


@pytest.mark.slow
def test_run_fraud_ensemble_reports_distribution() -> None:
    si = build_bulgaria_simulation_input(**BUILD_KW)
    spec = PerturbationSpec(
        ptype=PerturbationType.VOLUME_SPIKE, n_targets=3, intensity=8.0, seed=7
    )
    res = run_fraud_ensemble(
        si.engine_input, spec, seeds=[1, 2, 3], ticks=200, z_threshold=3.0
    )
    assert len(res.runs) == 3
    assert res.summary.n == 3
    assert res.seeds == [1, 2, 3]
    for stat in (res.summary.precision, res.summary.recall, res.summary.f1):
        assert 0.0 <= stat.mean <= 1.0
        assert stat.ci_low <= stat.mean <= stat.ci_high
    # The injected spike should be recovered on average.
    assert res.summary.recall.mean > 0.0


@pytest.mark.slow
def test_sweep_detection_threshold_is_monotone_in_recall() -> None:
    si = build_bulgaria_simulation_input(**BUILD_KW)
    spec = PerturbationSpec(
        ptype=PerturbationType.VOLUME_SPIKE, n_targets=3, intensity=10.0, seed=7
    )
    points = sweep_detection_threshold(
        si.engine_input,
        spec,
        thresholds=[2.0, 3.0, 5.0, 8.0],
        seeds=[1, 2],
        ticks=200,
    )
    assert [p.value for p in points] == [2.0, 3.0, 5.0, 8.0]
    recalls = [p.summary.recall.mean for p in points]
    # A stricter threshold can only flag fewer locations, so recall is non-increasing.
    assert all(a >= b - 1e-9 for a, b in zip(recalls, recalls[1:]))
