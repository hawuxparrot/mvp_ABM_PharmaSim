from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from compiler.types import EngineInput
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


# ---------------------------------------------------------------------------
# Engine event-log detectors (operate on the real ground-truth simulation log)
# ---------------------------------------------------------------------------

EventRecords = Sequence[Mapping[str, Any]]


def _location_org_types(inp: EngineInput) -> np.ndarray:
    return np.asarray(inp.org_type)[np.asarray(inp.location_org_id)]


def _zscore_flags(
    counts: dict[str, float], *, z_threshold: float
) -> dict[str, float]:
    if not counts:
        return {}
    vals = np.asarray(list(counts.values()), dtype=float)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma <= 1e-9:
        return {}
    return {
        k: (v - mu) / sigma
        for k, v in counts.items()
        if (v - mu) / sigma >= z_threshold
    }


def detect_volume_anomalies(
    records: EventRecords,
    inp: EngineInput,
    *,
    z_threshold: float = 3.0,
) -> dict[str, float]:
    """Flag locations whose total MOVE throughput (in + out) is a statistical outlier.

    Catches volume spikes and counterfeit injection, both of which surge the number of
    shipments through the affected nodes. Returns ``{location_ext_id: zscore}``.
    """
    throughput: dict[str, float] = defaultdict(float)
    for r in records:
        if r["event_type"] != "MOVE":
            continue
        src = r.get("from_location")
        dst = r.get("to_location")
        if src and src != "—":
            throughput[src] += 1.0
        if dst and dst != "—":
            throughput[dst] += 1.0
    return _zscore_flags(dict(throughput), z_threshold=z_threshold)


def _move_throughput(
    records: EventRecords, *, direction: str = "both"
) -> dict[str, float]:
    throughput: dict[str, float] = defaultdict(float)
    for r in records:
        if r["event_type"] != "MOVE":
            continue
        if direction in ("both", "out"):
            src = r.get("from_location")
            if src and src != "—":
                throughput[src] += 1.0
        if direction in ("both", "in"):
            dst = r.get("to_location")
            if dst and dst != "—":
                throughput[dst] += 1.0
    return dict(throughput)


def detect_volume_change(
    perturbed_records: EventRecords,
    baseline_records: EventRecords,
    inp: EngineInput,
    *,
    z_threshold: float = 3.0,
    direction: str = "out",
) -> dict[str, float]:
    """Flag locations whose MOVE throughput jumps versus a baseline run.

    A volume spike (or counterfeit injection) is a *change* in a node's traffic, not its
    absolute level -- structural hubs always carry the most traffic. Comparing against the
    baseline isolates the perturbed nodes. ``direction='out'`` (default) targets the surge
    *source* (where extra stock is injected and pushed downstream), avoiding flagging the
    downstream sinks the surge flows into. Returns ``{location_ext_id: zscore_of_delta}``.
    """
    tput_p = _move_throughput(perturbed_records, direction=direction)
    tput_b = _move_throughput(baseline_records, direction=direction)
    deltas = {
        loc: tput_p.get(loc, 0.0) - tput_b.get(loc, 0.0)
        for loc in set(tput_p) | set(tput_b)
    }
    return _zscore_flags(deltas, z_threshold=z_threshold)


def detect_cross_market_moves(
    records: EventRecords,
    inp: EngineInput,
) -> dict[str, float]:
    """Flag destination locations receiving packs whose origin market differs from the
    destination market (smuggling / cross-border diversion). Returns counts per location."""
    flagged: dict[str, float] = defaultdict(float)
    loc_market = np.asarray(inp.location_market_id)
    pack_origin = np.asarray(inp.pack_initial_market_id)
    for r in records:
        if r["event_type"] != "MOVE":
            continue
        to_id = r.get("to_location_id")
        pid = r.get("pack_id")
        if to_id is None or pid is None:
            continue
        if int(pack_origin[pid]) != int(loc_market[to_id]):
            flagged[r["to_location"]] += 1.0
    return dict(flagged)


def detect_theft(
    records: EventRecords,
    inp: EngineInput,
) -> dict[str, float]:
    """Flag shrinkage: DECOMMISSION events at supply nodes (wholesalers/OBP), which in a
    healthy chain only decommission (dispense) at terminals. Returns counts per location."""
    from policy.models import OrgType
    from compiler.enums import org_type_u8

    supply = {int(org_type_u8(OrgType.WHOLESALER)), int(org_type_u8(OrgType.OBP))}
    org_t = _location_org_types(inp)
    flagged: dict[str, float] = defaultdict(float)
    for r in records:
        if r["event_type"] != "DECOMMISSION":
            continue
        loc_id = r.get("from_location_id")
        if loc_id is None:
            continue
        if int(org_t[loc_id]) in supply:
            flagged[r["from_location"]] += 1.0
    return dict(flagged)


def evaluate_detection(
    predicted_ids: set[str],
    truth_ids: set[str],
) -> DetectionMetrics:
    """Score a set of predicted location ext-ids against ground-truth location ext-ids."""
    tp = len(predicted_ids & truth_ids)
    fp = len(predicted_ids - truth_ids)
    fn = len(truth_ids - predicted_ids)
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )
    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


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

