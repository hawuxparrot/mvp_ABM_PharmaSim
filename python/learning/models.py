from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class AlertCase:
    case_id: str
    tick: int
    pack_id: int | None
    event_type: str
    alert_code: str | None
    decommission_reason: str | None
    from_location_id: int | None
    to_location_id: int | None
    from_role: str | None
    to_role: str | None
    from_market: str | None
    to_market: str | None
    same_location: bool
    same_organization: bool
    same_city: bool
    same_postal_code: bool
    distance_km: float | None
    market_mismatch: bool
    prior_pack_events: int
    prior_pack_alerts: int
    source_recent_event_count: int
    destination_recent_event_count: int
    source_trust_score: float
    destination_trust_score: float


@dataclass(frozen=True)
class OperatorFeedback:
    label: bool
    confidence: float
    rule_id: str


@dataclass(frozen=True)
class CasePrediction:
    case_id: str
    probability: float
    prediction: bool
    queried: bool
    operator_label: bool | None = None
    operator_confidence: float | None = None
    operator_rule_id: str | None = None


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: payload[k] for k in allowed if k in payload})
