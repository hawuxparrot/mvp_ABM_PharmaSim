from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActorRole(StrEnum):
    PHARMACY = "PHARMACY"
    HOSPITAL = "HOSPITAL"
    WHOLESALER = "WHOLESALER"
    MANUFACTURER = "MANUFACTURER"
    REGULATOR = "REGULATOR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CanonicalNode:
    node_id: str
    name: str
    role: ActorRole
    market_code: str
    city: str
    address: str
    postal_code: str
    is_active: bool
    latitude: float | None = None
    longitude: float | None = None
    source_ids: dict[str, str] = field(default_factory=dict)
    quality_score: float = 1.0

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass(frozen=True)
class CanonicalEdge:
    src_node_id: str
    dst_node_id: str
    distance_km: float
    travel_minutes: float
    route_cost: float
    capacity_per_tick: int


@dataclass(frozen=True)
class CanonicalDataset:
    nodes: list[CanonicalNode]
    edges: list[CanonicalEdge]
    unresolved_geocodes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

