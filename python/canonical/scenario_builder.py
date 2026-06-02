from __future__ import annotations

from collections import Counter

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

from .models import ActorRole, CanonicalDataset


def _org_type_for_role(role: ActorRole) -> OrgType:
    if role == ActorRole.MANUFACTURER:
        return OrgType.OBP
    if role == ActorRole.WHOLESALER:
        return OrgType.WHOLESALER
    return OrgType.LOCAL_ORG


def build_scenario_from_canonical(
    dataset: CanonicalDataset,
    *,
    seed: int = 42,
    packs_per_obp: int = 250,
    max_locations: int = 350,
) -> Scenario:
    nodes = [n for n in dataset.nodes if n.is_active]
    if max_locations > 0 and len(nodes) > max_locations:
        # Keep a role-balanced deterministic slice so compile-time all-pairs routing remains tractable.
        by_role: dict[ActorRole, list] = {}
        for node in nodes:
            by_role.setdefault(node.role, []).append(node)
        ordered_roles = [
            ActorRole.MANUFACTURER,
            ActorRole.WHOLESALER,
            ActorRole.HOSPITAL,
            ActorRole.PHARMACY,
            ActorRole.REGULATOR,
            ActorRole.UNKNOWN,
        ]
        trimmed: list = []
        per_role_budget = max(1, max_locations // max(1, len(ordered_roles)))
        for role in ordered_roles:
            role_nodes = sorted(by_role.get(role, []), key=lambda n: n.node_id)
            trimmed.extend(role_nodes[:per_role_budget])
        if len(trimmed) < max_locations:
            seen = {n.node_id for n in trimmed}
            for node in sorted(nodes, key=lambda n: n.node_id):
                if node.node_id in seen:
                    continue
                trimmed.append(node)
                if len(trimmed) >= max_locations:
                    break
        nodes = trimmed[:max_locations]
    orgs: list[Organization] = []
    locs: list[Location] = []
    behaviors: dict[str, LocationBehavior] = {}

    for node in nodes:
        org_id = f"org_{node.node_id}"
        loc_id = f"loc_{node.node_id}"
        org_t = _org_type_for_role(node.role)
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

    node_to_loc = {n.node_id: f"loc_{n.node_id}" for n in nodes}
    edges: list[LocationEdge] = []
    for edge in dataset.edges:
        src = node_to_loc.get(edge.src_node_id)
        dst = node_to_loc.get(edge.dst_node_id)
        if src is None or dst is None:
            continue
        edges.append(
            LocationEdge(
                src_location_ext_id=src,
                dst_location_ext_id=dst,
                cost=float(max(0.01, edge.route_cost)),
                capacity=max(1, int(edge.capacity_per_tick)),
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

    obp_loc_ids: list[str] = []
    for node in nodes:
        if _org_type_for_role(node.role) == OrgType.OBP:
            obp_loc_ids.append(f"loc_{node.node_id}")
    if not obp_loc_ids:
        # No explicit manufacturer in sources; promote a few wholesalers to seed flow.
        for node in nodes:
            if _org_type_for_role(node.role) == OrgType.WHOLESALER:
                obp_loc_ids.append(f"loc_{node.node_id}")
            if len(obp_loc_ids) >= 3:
                break

    if not obp_loc_ids and locs:
        obp_loc_ids.append(locs[0].ext_id)

    batch = Batch(
        ext_id="batch_bg_registry_001",
        product_ext_id=product.ext_id,
        manufacturer_org_ext_id=orgs[0].ext_id if orgs else "org_missing",
        intended_markets=("BG",),
    )
    batches = [batch]

    packs: list[Pack] = []
    serial_counter = 0
    for loc_id in obp_loc_ids:
        for _ in range(packs_per_obp):
            serial_counter += 1
            packs.append(
                Pack(
                    ext_id=f"pack_bg_{serial_counter}",
                    product_ext_id=product.ext_id,
                    batch_ext_id=batch.ext_id,
                    serial=f"BGSN{serial_counter:09d}",
                    initial_market_code="BG",
                    initial_location_ext_id=loc_id,
                    initial_state=PackState.UPLOADED,
                )
            )

    if not orgs or not locs or not packs:
        counts = Counter(_org_type_for_role(node.role).value for node in nodes)
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

