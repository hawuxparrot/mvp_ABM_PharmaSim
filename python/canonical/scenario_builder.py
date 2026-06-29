from __future__ import annotations

from collections import Counter, deque

from policy.models import (
    Batch,
    Location,
    LocationBehavior,
    LocationEdge,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario,
)

from .models import ActorRole, CanonicalDataset, CanonicalNode
from .routing import build_supply_chain_edges, haversine_km


def _org_type_for_role(role: ActorRole) -> OrgType:
    if role == ActorRole.MANUFACTURER:
        return OrgType.OBP
    if role == ActorRole.WHOLESALER:
        return OrgType.WHOLESALER
    return OrgType.LOCAL_ORG


def _designate_obp_hubs(nodes: list[CanonicalNode], n_hubs: int) -> set[str]:
    """Pick import/OBP hubs, dispersed across the country.

    If the data has explicit manufacturers, they are the OBPs. Bulgaria's registry has
    none, so we promote wholesalers to act as import hubs. Hubs are chosen by
    farthest-point sampling (seeded at the most central wholesaler) so they spread across
    regions instead of clustering in the capital -- this matters because each supply node
    feeds a single preferred downstream edge, so clustered hubs would all funnel into the
    same wholesaler and collapse the distribution graph to one chain.
    """
    coord_nodes = [n for n in nodes if n.has_coordinates and n.is_active]
    manufacturers = [n for n in coord_nodes if n.role == ActorRole.MANUFACTURER]
    if manufacturers:
        return {n.node_id for n in manufacturers}

    wholesalers = [n for n in coord_nodes if n.role == ActorRole.WHOLESALER]
    if not wholesalers:
        return set()

    n_hubs = max(1, min(n_hubs, len(wholesalers)))
    lat0 = sum(float(n.latitude) for n in coord_nodes) / len(coord_nodes)  # type: ignore[arg-type]
    lon0 = sum(float(n.longitude) for n in coord_nodes) / len(coord_nodes)  # type: ignore[arg-type]

    def dist_to(a: CanonicalNode, lat: float, lon: float) -> float:
        assert a.latitude is not None and a.longitude is not None
        return haversine_km(lat, lon, a.latitude, a.longitude)

    seed = min(wholesalers, key=lambda n: (dist_to(n, lat0, lon0), n.node_id))
    chosen: list[CanonicalNode] = [seed]
    chosen_ids = {seed.node_id}
    while len(chosen) < n_hubs:
        best: CanonicalNode | None = None
        best_key: tuple[float, str] | None = None
        for cand in wholesalers:
            if cand.node_id in chosen_ids:
                continue
            assert cand.latitude is not None and cand.longitude is not None
            min_d = min(
                dist_to(cand, float(c.latitude), float(c.longitude))  # type: ignore[arg-type]
                for c in chosen
            )
            # Maximize distance to the nearest already-chosen hub (tie-break deterministic).
            key = (-min_d, cand.node_id)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        if best is None:
            break
        chosen.append(best)
        chosen_ids.add(best.node_id)
    return chosen_ids


def _select_connected_nodes(
    nodes: list[CanonicalNode],
    edges: list[tuple[str, str]],
    obp_node_ids: set[str],
    max_locations: int,
) -> list[CanonicalNode]:
    """Connectivity-first node selection.

    BFS outward from the OBP hubs across the directed supply graph so that every kept
    node is reachable from a hub (and edges between kept nodes survive). If budget
    remains, fill with the remaining nodes (coordinate-bearing first) so the map stays
    populated. This replaces the old role/id-order trim that dropped the very endpoints
    that formed the graph, yielding 0 edges.
    """
    node_by_id = {n.node_id: n for n in nodes}
    adj: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for src, dst in edges:
        if src in adj and dst in node_by_id:
            adj[src].append(dst)

    cap = max_locations if max_locations > 0 else len(nodes)

    selected: list[str] = []
    seen: set[str] = set()

    def add(nid: str) -> bool:
        if nid in seen or nid not in node_by_id:
            return False
        if len(selected) >= cap:
            return False
        selected.append(nid)
        seen.add(nid)
        return True

    queue: deque[str] = deque()
    for nid in sorted(obp_node_ids):
        if add(nid):
            queue.append(nid)

    while queue and len(selected) < cap:
        cur = queue.popleft()
        for nxt in adj.get(cur, []):
            if add(nxt):
                queue.append(nxt)

    if len(selected) < cap:
        remaining = [n for n in nodes if n.node_id not in seen]
        remaining.sort(key=lambda n: (not n.has_coordinates, n.node_id))
        for n in remaining:
            if not add(n.node_id):
                break

    return [node_by_id[nid] for nid in selected]


def build_scenario_from_canonical(
    dataset: CanonicalDataset,
    *,
    seed: int = 42,
    packs_per_obp: int = 1000,
    max_locations: int = 8192,
    n_obp_hubs: int = 4,
    packs_per_wholesaler: int = 50,
) -> Scenario:
    active = [n for n in dataset.nodes if n.is_active]

    obp_node_ids = _designate_obp_hubs(active, n_obp_hubs)

    canonical_edges = build_supply_chain_edges(active, obp_node_ids)
    edge_pairs: list[tuple[str, str]] = [
        (e.src_node_id, e.dst_node_id) for e in canonical_edges
    ]

    nodes = _select_connected_nodes(active, edge_pairs, obp_node_ids, max_locations)
    selected_ids = {n.node_id for n in nodes}

    orgs: list[Organization] = []
    locs: list[Location] = []
    behaviors: dict[str, LocationBehavior] = {}

    def org_type_for_node(node: CanonicalNode) -> OrgType:
        if node.node_id in obp_node_ids:
            return OrgType.OBP
        return _org_type_for_role(node.role)

    for node in nodes:
        org_id = f"org_{node.node_id}"
        loc_id = f"loc_{node.node_id}"
        org_t = org_type_for_node(node)
        orgs.append(Organization(ext_id=org_id, org_type=org_t))
        locs.append(
            Location(
                ext_id=loc_id,
                org_ext_id=org_id,
                market_code=node.market_code,
                postal_code=node.postal_code or "0000",
            )
        )
        if org_t == OrgType.LOCAL_ORG:
            behaviors[loc_id] = LocationBehavior(
                verify_prob=0.12,
                decomission_prob=0.03,
                reactivate_prob=0.002,
            )

    edges: list[LocationEdge] = []
    for e in canonical_edges:
        if e.src_node_id in selected_ids and e.dst_node_id in selected_ids:
            edges.append(
                LocationEdge(
                    src_location_ext_id=f"loc_{e.src_node_id}",
                    dst_location_ext_id=f"loc_{e.dst_node_id}",
                    cost=float(max(0.01, e.route_cost)),
                    capacity=max(1, int(e.capacity_per_tick)),
                )
            )

    product = Product(
        ext_id="prod_bg_registry",
        codes=(
            ProductCode(
                scheme=ProductCodeScheme.GTIN,
                value="00000000000001",
                is_primary=True,
            ),
        ),
    )
    products = [product]

    obp_loc_ids = [f"loc_{n.node_id}" for n in nodes if n.node_id in obp_node_ids]
    wholesaler_loc_ids = [
        f"loc_{n.node_id}"
        for n in nodes
        if n.node_id not in obp_node_ids and org_type_for_node(n) == OrgType.WHOLESALER
    ]
    if not obp_loc_ids:
        # No hub survived selection (e.g. tiny/edgeless dataset); seed at the first loc.
        if wholesaler_loc_ids:
            obp_loc_ids.append(wholesaler_loc_ids[0])
        elif locs:
            obp_loc_ids.append(locs[0].ext_id)

    batch = Batch(
        ext_id="batch_bg_registry_001",
        product_ext_id=product.ext_id,
        manufacturer_org_ext_id=orgs[0].ext_id if orgs else "org_missing",
        intended_markets=("BG",),
    )
    batches = [batch]

    # Seed inventory at supply nodes. Packs are ACTIVE because they represent stock already
    # circulating in distribution; the supply phase ships them downstream and they
    # decommission (are dispensed) at pharmacies/hospitals. Stocking every wholesaler -- not
    # only the import hubs -- gives broad terminal coverage and a baseline traffic
    # population the anomaly detectors can score against.
    packs: list[Pack] = []
    serial_counter = 0

    def seed_packs(loc_id: str, count: int) -> None:
        nonlocal serial_counter
        for _ in range(count):
            serial_counter += 1
            packs.append(
                Pack(
                    ext_id=f"pack_bg_{serial_counter}",
                    product_ext_id=product.ext_id,
                    batch_ext_id=batch.ext_id,
                    serial=f"BGSN{serial_counter:09d}",
                    initial_market_code="BG",
                    initial_location_ext_id=loc_id,
                    initial_state=PackState.ACTIVE,
                )
            )

    for loc_id in obp_loc_ids:
        seed_packs(loc_id, packs_per_obp)
    if packs_per_wholesaler > 0:
        for loc_id in wholesaler_loc_ids:
            seed_packs(loc_id, packs_per_wholesaler)

    if not orgs or not locs or not packs:
        counts = Counter(org_type_for_node(node).value for node in nodes)
        raise ValueError(
            "Cannot build scenario from canonical dataset; "
            f"orgs={len(orgs)} locs={len(locs)} packs={len(packs)} role_counts={dict(counts)}"
        )

    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location=behaviors,
        seed=seed,
    )
