from __future__ import annotations

from analytics.fraud import (
    detect_cross_market,
    detect_volume_spikes,
    evaluate_detector,
    inject_cross_market_anomalies,
    inject_volume_spike_anomalies,
)
from canonical.transactions import build_synthetic_transaction_plan, plan_stage_counts
from policy.scenarios import (
    bulgaria_registry_experiment_bundle,
    two_markets_demo,
)
from policy.transactions import AnomalyType, TransactionIntent, TransactionPlan, TransactionStage


def test_synthetic_transaction_plan_has_lifecycle_balance() -> None:
    plan = build_synthetic_transaction_plan(
        two_markets_demo(),
        horizon_ticks=20,
        order_lambda_per_edge=0.9,
        seed=7,
    )
    counts = plan_stage_counts(plan)
    assert counts.get(TransactionStage.ORDER_CREATED.value, 0) > 0
    assert counts[TransactionStage.ORDER_CREATED.value] == counts[TransactionStage.SHIPMENT_DISPATCHED.value]
    assert counts[TransactionStage.ORDER_CREATED.value] == counts[TransactionStage.SHIPMENT_DELIVERED.value]


def test_inject_volume_spike_anomalies_changes_units_and_labels() -> None:
    base = build_synthetic_transaction_plan(
        two_markets_demo(),
        horizon_ticks=12,
        order_lambda_per_edge=0.8,
        seed=3,
    )
    mutated, labels = inject_volume_spike_anomalies(base, fraction=0.2, factor=8.0, seed=11)
    assert labels
    assert all(l.anomaly_type == AnomalyType.VOLUME_SPIKE for l in labels)

    before = {t.tx_id: t.units for t in base.intents}
    after = {t.tx_id: t.units for t in mutated.intents}
    assert any(after[k] > before[k] for k in before.keys() if k in after)


def test_cross_market_injection_and_detector_pair() -> None:
    base = build_synthetic_transaction_plan(
        two_markets_demo(),
        horizon_ticks=14,
        order_lambda_per_edge=1.0,
        seed=9,
    )
    mutated, labels = inject_cross_market_anomalies(base, fraction=0.25, seed=5)
    predicted = detect_cross_market(mutated)
    metrics = evaluate_detector(predicted, labels, anomaly_type=AnomalyType.CROSS_MARKET)
    # Detector is simple equality check; should catch injected market mismatches reliably.
    assert metrics.recall >= 0.9


def test_volume_spike_detector_catches_obvious_outlier() -> None:
    base = TransactionPlan(
        intents=[
            TransactionIntent(
                tx_id=f"tx_{i}",
                tick=0,
                stage=TransactionStage.SHIPMENT_DISPATCHED,
                src_location_ext_id="s",
                dst_location_ext_id="d",
                src_market_code="BG",
                dst_market_code="BG",
                product_ext_id="p",
                units=10 if i < 20 else 300,
                route_cost_hint=1.0,
            )
            for i in range(21)
        ]
    )
    predicted = detect_volume_spikes(base, z_threshold=3.0)
    assert "tx_20" in predicted


def test_bulgaria_experiment_bundle_smoke() -> None:
    bundle = bulgaria_registry_experiment_bundle(
        geocode=False,
        max_locations=80,
        packs_per_obp=20,
        tx_horizon_ticks=10,
        tx_order_lambda_per_edge=0.5,
        tx_max_units_per_order=20,
    )
    assert bundle.scenario.locations
    assert bundle.transaction_plan.intents
    assert bundle.anomaly_labels

