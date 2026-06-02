from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy.transactions import (
    AnomalyLabel,
    AnomalyType,
    TransactionIntent,
    TransactionPlan,
    TransactionStage,
)


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def inject_volume_spike_anomalies(
    plan: TransactionPlan,
    *,
    fraction: float = 0.03,
    factor: float = 6.0,
    seed: int = 42,
) -> tuple[TransactionPlan, list[AnomalyLabel]]:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if factor <= 1.0:
        raise ValueError("factor must be > 1")

    rng = np.random.default_rng(seed)
    intents = list(plan.intents)
    candidates = [
        i
        for i, t in enumerate(intents)
        if t.stage == TransactionStage.SHIPMENT_DISPATCHED
    ]
    if not candidates:
        return plan, []

    n = max(1, int(round(len(candidates) * fraction)))
    chosen_ix = set(int(x) for x in rng.choice(candidates, size=min(n, len(candidates)), replace=False))

    labels: list[AnomalyLabel] = []
    out: list[TransactionIntent] = []
    for i, intent in enumerate(intents):
        if i not in chosen_ix:
            out.append(intent)
            continue
        new_units = max(intent.units + 1, int(round(intent.units * factor)))
        mutated = TransactionIntent(
            tx_id=intent.tx_id,
            tick=intent.tick,
            stage=intent.stage,
            src_location_ext_id=intent.src_location_ext_id,
            dst_location_ext_id=intent.dst_location_ext_id,
            src_market_code=intent.src_market_code,
            dst_market_code=intent.dst_market_code,
            product_ext_id=intent.product_ext_id,
            units=new_units,
            route_cost_hint=intent.route_cost_hint,
            metadata={**intent.metadata, "anomaly_injected": "volume_spike"},
        )
        out.append(mutated)
        labels.append(
            AnomalyLabel(
                tx_id=intent.tx_id,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                reason="Injected multiplier on shipment-dispatch units",
                severity=float(new_units / max(1, intent.units)),
            )
        )
    return TransactionPlan(intents=out), labels


def inject_cross_market_anomalies(
    plan: TransactionPlan,
    *,
    fraction: float = 0.02,
    seed: int = 43,
) -> tuple[TransactionPlan, list[AnomalyLabel]]:
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in [0, 1]")

    rng = np.random.default_rng(seed)
    intents = list(plan.intents)
    markets = sorted({i.src_market_code for i in intents if i.src_market_code != "UNK"})
    if len(markets) < 2:
        return plan, []

    candidates = [
        i
        for i, t in enumerate(intents)
        if t.stage == TransactionStage.SHIPMENT_DISPATCHED
    ]
    if not candidates:
        return plan, []

    n = max(1, int(round(len(candidates) * fraction)))
    chosen_ix = set(int(x) for x in rng.choice(candidates, size=min(n, len(candidates)), replace=False))

    labels: list[AnomalyLabel] = []
    out: list[TransactionIntent] = []
    for i, intent in enumerate(intents):
        if i not in chosen_ix:
            out.append(intent)
            continue
        alternatives = [m for m in markets if m != intent.src_market_code]
        if not alternatives:
            out.append(intent)
            continue
        wrong_market = str(rng.choice(alternatives))
        mutated = TransactionIntent(
            tx_id=intent.tx_id,
            tick=intent.tick,
            stage=intent.stage,
            src_location_ext_id=intent.src_location_ext_id,
            dst_location_ext_id=intent.dst_location_ext_id,
            src_market_code=intent.src_market_code,
            dst_market_code=wrong_market,
            product_ext_id=intent.product_ext_id,
            units=intent.units,
            route_cost_hint=intent.route_cost_hint,
            metadata={**intent.metadata, "anomaly_injected": "cross_market"},
        )
        out.append(mutated)
        labels.append(
            AnomalyLabel(
                tx_id=intent.tx_id,
                anomaly_type=AnomalyType.CROSS_MARKET,
                reason="Injected destination market mismatch",
                severity=1.0,
            )
        )
    return TransactionPlan(intents=out), labels


def detect_volume_spikes(
    plan: TransactionPlan,
    *,
    z_threshold: float = 3.0,
) -> dict[str, float]:
    """
    Baseline detector: shipment-dispatch transactions with high z-score on units.
    """
    dispatch = [t for t in plan.intents if t.stage == TransactionStage.SHIPMENT_DISPATCHED]
    if not dispatch:
        return {}
    units = np.asarray([t.units for t in dispatch], dtype=float)
    mu = float(units.mean())
    sigma = float(units.std())
    if sigma <= 1e-6:
        return {}
    out: dict[str, float] = {}
    for t in dispatch:
        z = (float(t.units) - mu) / sigma
        if z >= z_threshold:
            out[t.tx_id] = z
    return out


def detect_cross_market(
    plan: TransactionPlan,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in plan.intents:
        if t.stage != TransactionStage.SHIPMENT_DISPATCHED:
            continue
        if t.src_market_code != "UNK" and t.dst_market_code != "UNK":
            if t.src_market_code != t.dst_market_code:
                out[t.tx_id] = 1.0
    return out


def evaluate_detector(
    predicted: dict[str, float],
    labels: list[AnomalyLabel],
    *,
    anomaly_type: AnomalyType | None = None,
) -> DetectionMetrics:
    truth = {
        l.tx_id
        for l in labels
        if anomaly_type is None or l.anomaly_type == anomaly_type
    }
    pred = set(predicted.keys())
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )

