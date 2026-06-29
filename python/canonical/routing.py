from __future__ import annotations

import math

from .models import ActorRole, CanonicalEdge, CanonicalNode


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _is_supply_role(role: ActorRole) -> bool:
    return role in {ActorRole.MANUFACTURER, ActorRole.WHOLESALER}


def _is_terminal_role(role: ActorRole) -> bool:
    return role in {ActorRole.PHARMACY, ActorRole.HOSPITAL}


def _is_supply_chain_terminal_role(role: ActorRole) -> bool:
    """Last-mile destinations for the tiered supply graph.

    Includes ``UNKNOWN`` because those nodes map to ``LOCAL_ORG`` (pharmacy/hospital-like)
    terminals in the scenario, and many carry coordinates even before the full pharmacy
    geocode completes, keeping the distribution chain functional.
    """
    return role in {ActorRole.PHARMACY, ActorRole.HOSPITAL, ActorRole.UNKNOWN}


def _capacity_for_roles(src: ActorRole, dst: ActorRole) -> int:
    if src == ActorRole.MANUFACTURER and dst == ActorRole.WHOLESALER:
        return 1200
    if src == ActorRole.WHOLESALER and dst == ActorRole.PHARMACY:
        return 300
    if src == ActorRole.WHOLESALER and dst == ActorRole.HOSPITAL:
        return 500
    return 100


def _edge_between(
    src: CanonicalNode,
    dst: CanonicalNode,
    distance: float,
    *,
    speed_kmph: float,
    cost_per_km: float,
) -> CanonicalEdge:
    travel_minutes = (distance / max(1e-6, speed_kmph)) * 60.0
    return CanonicalEdge(
        src_node_id=src.node_id,
        dst_node_id=dst.node_id,
        distance_km=distance,
        travel_minutes=travel_minutes,
        route_cost=max(0.01, distance * cost_per_km),
        capacity_per_tick=_capacity_for_roles(src.role, dst.role),
    )


def _nearest(
    src: CanonicalNode,
    candidates: list[CanonicalNode],
    *,
    k: int,
    max_distance_km: float,
) -> list[tuple[float, CanonicalNode]]:
    assert src.latitude is not None and src.longitude is not None
    scored: list[tuple[float, CanonicalNode]] = []
    for dst in candidates:
        if dst.node_id == src.node_id:
            continue
        assert dst.latitude is not None and dst.longitude is not None
        d = haversine_km(src.latitude, src.longitude, dst.latitude, dst.longitude)
        if d <= max_distance_km:
            scored.append((d, dst))
    scored.sort(key=lambda x: x[0])
    return scored[:k]


def build_supply_chain_edges(
    nodes: list[CanonicalNode],
    obp_node_ids: set[str],
    *,
    k_hub_to_wholesaler: int = 4,
    k_wholesaler_to_terminal: int = 6,
    hub_max_distance_km: float = 600.0,
    last_mile_max_distance_km: float = 120.0,
    speed_kmph: float = 55.0,
    cost_per_km: float = 0.8,
) -> list[CanonicalEdge]:
    """Build a tiered distribution graph: OBP/import-hub -> wholesaler -> terminal.

    The two-tier topology is what lets packs become ``ACTIVE`` (on arrival at a
    wholesaler) before reaching a pharmacy/hospital, so downstream verify/decommission
    behavior actually fires. Edges are emitted grouped by source so the compiler's
    "first outgoing edge = preferred supplier" rule yields a coherent hierarchy:
    every hub's preferred edge points at a wholesaler and every wholesaler's preferred
    edge points at a terminal.
    """
    with_coords = [n for n in nodes if n.has_coordinates and n.is_active]
    wholesalers = [
        n
        for n in with_coords
        if n.role == ActorRole.WHOLESALER and n.node_id not in obp_node_ids
    ]
    terminals = [
        n
        for n in with_coords
        if _is_supply_chain_terminal_role(n.role) and n.node_id not in obp_node_ids
    ]

    edges: list[CanonicalEdge] = []

    # Tier 1: hubs -> nearest wholesalers (national distribution).
    for hub in with_coords:
        if hub.node_id not in obp_node_ids:
            continue
        targets = wholesalers if wholesalers else terminals
        for distance, dst in _nearest(
            hub, targets, k=k_hub_to_wholesaler, max_distance_km=hub_max_distance_km
        ):
            edges.append(
                _edge_between(
                    hub, dst, distance, speed_kmph=speed_kmph, cost_per_km=cost_per_km
                )
            )

    # Tier 2: wholesalers -> nearest terminals (last mile).
    for wh in wholesalers:
        for distance, dst in _nearest(
            wh,
            terminals,
            k=k_wholesaler_to_terminal,
            max_distance_km=last_mile_max_distance_km,
        ):
            edges.append(
                _edge_between(
                    wh, dst, distance, speed_kmph=speed_kmph, cost_per_km=cost_per_km
                )
            )

    return edges


def build_routing_edges(
    nodes: list[CanonicalNode],
    *,
    k_neighbors: int = 3,
    max_distance_km: float = 300.0,
    speed_kmph: float = 55.0,
    cost_per_km: float = 0.8,
) -> list[CanonicalEdge]:
    with_coords = [n for n in nodes if n.has_coordinates and n.is_active]
    if not with_coords:
        return []

    edges: list[CanonicalEdge] = []
    for src in with_coords:
        if not _is_supply_role(src.role):
            continue
        candidates: list[tuple[float, CanonicalNode]] = []
        for dst in with_coords:
            if src.node_id == dst.node_id:
                continue
            if not _is_terminal_role(dst.role) and dst.role != ActorRole.WHOLESALER:
                continue
            if src.role == ActorRole.WHOLESALER and dst.role == ActorRole.WHOLESALER:
                continue
            assert src.latitude is not None and src.longitude is not None
            assert dst.latitude is not None and dst.longitude is not None
            distance = haversine_km(src.latitude, src.longitude, dst.latitude, dst.longitude)
            if distance <= max_distance_km:
                candidates.append((distance, dst))
        candidates.sort(key=lambda x: x[0])
        for distance, dst in candidates[:k_neighbors]:
            travel_minutes = (distance / max(1e-6, speed_kmph)) * 60.0
            edges.append(
                CanonicalEdge(
                    src_node_id=src.node_id,
                    dst_node_id=dst.node_id,
                    distance_km=distance,
                    travel_minutes=travel_minutes,
                    route_cost=distance * cost_per_km,
                    capacity_per_tick=_capacity_for_roles(src.role, dst.role),
                )
            )
    return edges

