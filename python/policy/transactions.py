from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TransactionStage(StrEnum):
    ORDER_CREATED = "ORDER_CREATED"
    SHIPMENT_DISPATCHED = "SHIPMENT_DISPATCHED"
    SHIPMENT_DELIVERED = "SHIPMENT_DELIVERED"


class AnomalyType(StrEnum):
    VOLUME_SPIKE = "VOLUME_SPIKE"
    CROSS_MARKET = "CROSS_MARKET"


@dataclass(frozen=True)
class TransactionIntent:
    tx_id: str
    tick: int
    stage: TransactionStage
    src_location_ext_id: str
    dst_location_ext_id: str
    src_market_code: str
    dst_market_code: str
    product_ext_id: str
    units: int
    route_cost_hint: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TransactionPlan:
    intents: list[TransactionIntent]


@dataclass(frozen=True)
class AnomalyLabel:
    tx_id: str
    anomaly_type: AnomalyType
    reason: str
    severity: float

