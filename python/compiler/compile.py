"""Compile a validated :class:`~policy.models.Scenario` into :class:`~compiler.types.EngineInput`."""

from __future__ import annotations

import numpy as np
import heapq
import math

from compiler.enums import org_type_u8, pack_state_u8
from compiler.types import ENGINE_INPUT_SCHEMA_VERSION, EngineInput
from compiler.validate import validate_scenario
from policy.models import OrgType, Scenario


def _collect_market_codes(s: Scenario) -> list[str]:
    codes: set[str] = set()
    for loc in s.locations:
        codes.add(loc.market_code)
    for pk in s.packs:
        codes.add(pk.initial_market_code)
    for b in s.batches:
        for m in b.intended_markets:
            codes.add(m)
    return sorted(codes)

def _build_multihop_route(
    n_loc: int,
    n_edge: int,
    edge_src_location_id: np.ndarray,
    edge_dst_location_id: np.ndarray,
    edge_cost: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # adjacency: src -> list[(dst, edge_id, weight)]
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n_loc)]
    for edge_id in range(len(edge_src_location_id)):
        src_id = int(edge_src_location_id[edge_id])
        dst_id = int(edge_dst_location_id[edge_id])
        cost = float(edge_cost[edge_id])
        adj[src_id].append((dst_id, edge_id, cost))

    off = np.zeros(n_loc + 1, dtype=np.uint32)
    flat_dst: list[int] = []
    flat_next_edge: list[int] = []

    for src in range(n_loc):
        dist = [math.inf] * n_loc
        parent = [-1] * n_loc   # prev node
        parent_edge = [-1] * n_loc   # edge used to arrive at curr node
        dist[src] = 0.0
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, eid, weight in adj[u]:
                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    parent_edge[v] = eid
                    heapq.heappush(pq, (dist[v], v))
        # emit routes for reachable dst != src
        for dst in range(n_loc):
            if dst == src or not math.isfinite(dist[dst]):
                continue
            # walk back dst->src and recover first edge out of src
            cur = dst
            first_edge = -1
            while parent[cur] != -1:
                e = parent_edge[cur]
                p = parent[cur]
                first_edge = e
                cur = p
                if cur == src:
                    break
            if cur != src or first_edge < 0:
                continue
            flat_dst.append(dst)
            flat_next_edge.append(first_edge)
        off[src + 1] = len(flat_dst)
    return (
        off,
        np.array(flat_dst, dtype=np.uint32),
        np.array(flat_next_edge, dtype=np.uint32),
    )


def compile_scenario(scenario: Scenario) -> EngineInput:
    """
    Validate *scenario*, assign dense IDs (list order = row id), build columnar :class:`EngineInput`.

    ID policy: for each entity list on :class:`~policy.models.Scenario`, ``row_index`` is the dense
    id (0 .. n-1). Markets are interned: sorted unique market strings → ``market_id``.

    Location behavior (Option B): always emit dense arrays of length ``n_locations``; rows are
    zero unless :attr:`~policy.models.Scenario.behavior_by_location` defines a row for that site.
    """
    s = validate_scenario(scenario)

    market_code = _collect_market_codes(s)
    market_str_to_id = {m: i for i, m in enumerate(market_code)}
    n_markets = len(market_code)

    org_ext_to_id = {o.ext_id: i for i, o in enumerate(s.organizations)}
    loc_ext_to_id = {loc.ext_id: i for i, loc in enumerate(s.locations)}
    product_ext_to_id = {p.ext_id: i for i, p in enumerate(s.products)}
    batch_ext_to_id = {b.ext_id: i for i, b in enumerate(s.batches)}

    n_org = len(s.organizations)
    n_loc = len(s.locations)
    n_prod = len(s.products)
    n_batch = len(s.batches)
    n_pack = len(s.packs)
    n_edge = len(s.location_edges)

    org_type = np.zeros(n_org, dtype=np.uint8)
    org_ext_id = [o.ext_id for o in s.organizations]
    for i, o in enumerate(s.organizations):
        org_type[i] = org_type_u8(o.org_type)

    location_org_id = np.zeros(n_loc, dtype=np.uint32)
    location_market_id = np.zeros(n_loc, dtype=np.uint32)
    location_ext_id = [loc.ext_id for loc in s.locations]
    for i, loc in enumerate(s.locations):
        location_org_id[i] = org_ext_to_id[loc.org_ext_id]
        location_market_id[i] = market_str_to_id[loc.market_code]

    edge_src_location_id = np.zeros(n_edge, dtype=np.uint32)
    edge_dst_location_id = np.zeros(n_edge, dtype=np.uint32)
    edge_cost = np.zeros(n_edge, dtype=np.float32)
    edge_capacity = np.zeros(n_edge, dtype=np.uint32)
    for i, edge in enumerate(s.location_edges):
        edge_src_location_id[i] = loc_ext_to_id[edge.src_location_ext_id]
        edge_dst_location_id[i] = loc_ext_to_id[edge.dst_location_ext_id]
        edge_cost[i] = np.float32(edge.cost)
        edge_capacity[i] = np.uint32(edge.capacity)

    out_edges_by_location: list[list[int]] = [[] for _ in range(n_loc)]
    for edge_id in range(n_edge):
        src_id = int(edge_src_location_id[edge_id])
        out_edges_by_location[src_id].append(edge_id)

    location_out_edge_offset = np.zeros(n_loc + 1, dtype=np.uint32)
    out_edge_flat: list[int] = []
    for loc_id, out_ids in enumerate(out_edges_by_location):
        out_edge_flat.extend(out_ids)
        location_out_edge_offset[loc_id + 1] = len(out_edge_flat)
    location_out_edge_id = np.array(out_edge_flat, dtype=np.uint32)

    location_route_offset, location_route_dst_location_id, location_route_next_edge_id = _build_multihop_route(
        n_loc, n_edge, edge_src_location_id, edge_dst_location_id, edge_cost
    )

    batch_product_id = np.zeros(n_batch, dtype=np.uint32)
    batch_manufacturer_org_id = np.zeros(n_batch, dtype=np.uint32)
    batch_ext_id = [b.ext_id for b in s.batches]
    flat_markets: list[int] = []
    offset: list[int] = [0]
    for b in s.batches:
        for m in b.intended_markets:
            flat_markets.append(market_str_to_id[m])
        offset.append(len(flat_markets))

    batch_intended_market_offset = np.array(offset, dtype=np.uint32)
    batch_intended_market_id = np.array(flat_markets, dtype=np.uint32)

    for i, b in enumerate(s.batches):
        batch_product_id[i] = product_ext_to_id[b.product_ext_id]
        batch_manufacturer_org_id[i] = org_ext_to_id[b.manufacturer_org_ext_id]

    pack_product_id = np.zeros(n_pack, dtype=np.uint32)
    pack_batch_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_location_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_market_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_state = np.zeros(n_pack, dtype=np.uint8)
    pack_serial: list[str] = []
    pack_ext_id = [p.ext_id for p in s.packs]

    for i, p in enumerate(s.packs):
        pack_product_id[i] = product_ext_to_id[p.product_ext_id]
        pack_batch_id[i] = batch_ext_to_id[p.batch_ext_id]
        pack_initial_location_id[i] = loc_ext_to_id[p.initial_location_ext_id]
        pack_initial_market_id[i] = market_str_to_id[p.initial_market_code]
        pack_initial_state[i] = pack_state_u8(p.initial_state)
        pack_serial.append(p.serial)

    location_has_behavior = np.zeros(n_loc, dtype=np.uint8)
    location_verify_prob = np.zeros(n_loc, dtype=np.float32)
    location_decommission_prob = np.zeros(n_loc, dtype=np.float32)
    location_reactivate_prob = np.zeros(n_loc, dtype=np.float32)
    location_demand_policy_id = np.zeros(n_loc, dtype=np.uint8)
    location_supply_policy_id = np.zeros(n_loc, dtype=np.uint8)
    location_initial_on_hand = np.zeros(n_loc, dtype=np.int32)
    location_initial_backlog = np.zeros(n_loc, dtype=np.int32)
    location_initial_pipeline_outstanding = np.zeros(n_loc, dtype=np.int32)
    location_demand_const_rate = np.zeros(n_loc, dtype=np.int32)
    location_reorder_point_s = np.zeros(n_loc, dtype=np.int32)
    location_order_up_to_S = np.zeros(n_loc, dtype=np.int32)
    location_unfulfilled_unit_penalty = np.zeros(n_loc, dtype=np.float32)
    location_preferred_supplier_edge_id = np.full(n_loc, np.uint32(2**32 - 1), dtype=np.uint32)
    location_base_stock_level = np.zeros(n_loc, dtype=np.int32)
    location_ewma_alpha = np.zeros(n_loc, dtype=np.float32)
    location_supply_capacity_per_tick = np.zeros(n_loc, dtype=np.uint32)
    location_min_order_interval_ticks = np.zeros(n_loc, dtype=np.uint32)
    edge_lead_time_ticks = np.zeros(n_edge, dtype=np.uint16)
    for i, loc in enumerate(s.locations):
        beh = s.behavior_by_location.get(loc.ext_id)
        if beh is None:
            continue
        location_has_behavior[i] = 1
        location_verify_prob[i] = np.float32(beh.verify_prob)
        location_decommission_prob[i] = np.float32(beh.decomission_prob)
        location_reactivate_prob[i] = np.float32(beh.reactivate_prob)
    
    LOCAL_ORG_U8 = int(org_type_u8(OrgType.LOCAL_ORG))
    WHOLESALER_U8 = int(org_type_u8(OrgType.WHOLESALER))
    for i in range(n_loc):
        org_row = int(location_org_id[i])
        ot = int(org_type[org_row])
        if ot == LOCAL_ORG_U8:
            location_demand_policy_id[i] = np.uint8(1)
            location_demand_const_rate[i] = np.int32(1)
        elif ot == WHOLESALER_U8:
            location_supply_policy_id[i] = np.uint8(1)
            location_supply_capacity_per_tick[i] = np.uint32(10)
            start = int(location_out_edge_offset[i])
            end = int(location_out_edge_offset[i + 1])
            if start < end:
                location_preferred_supplier_edge_id[i] = np.uint32(int(location_out_edge_id[start]))
    

    engine_input = EngineInput(
        schema_version=ENGINE_INPUT_SCHEMA_VERSION,
        seed=s.seed,
        n_organizations=n_org,
        n_locations=n_loc,
        n_products=n_prod,
        n_batches=n_batch,
        n_packs=n_pack,
        n_markets=n_markets,
        n_edges=n_edge,
        market_code=market_code,
        org_type=org_type,
        location_org_id=location_org_id,
        location_market_id=location_market_id,
        location_out_edge_offset=location_out_edge_offset,
        location_out_edge_id=location_out_edge_id,
        edge_src_location_id=edge_src_location_id,
        edge_dst_location_id=edge_dst_location_id,
        edge_cost=edge_cost,
        edge_capacity=edge_capacity,
        batch_product_id=batch_product_id,
        batch_manufacturer_org_id=batch_manufacturer_org_id,
        batch_intended_market_offset=batch_intended_market_offset,
        batch_intended_market_id=batch_intended_market_id,
        pack_product_id=pack_product_id,
        pack_batch_id=pack_batch_id,
        pack_initial_location_id=pack_initial_location_id,
        pack_initial_market_id=pack_initial_market_id,
        pack_initial_state=pack_initial_state,
        pack_serial=pack_serial,
        location_has_behavior=location_has_behavior,
        location_verify_prob=location_verify_prob,
        location_decommission_prob=location_decommission_prob,
        location_reactivate_prob=location_reactivate_prob,
        location_demand_policy_id=location_demand_policy_id,
        location_supply_policy_id=location_supply_policy_id,
        location_initial_on_hand=location_initial_on_hand,
        location_initial_backlog=location_initial_backlog,
        location_initial_pipeline_outstanding=location_initial_pipeline_outstanding,
        location_demand_const_rate=location_demand_const_rate,
        location_reorder_point_s=location_reorder_point_s,
        location_order_up_to_S=location_order_up_to_S,
        location_base_stock_level=location_base_stock_level,
        location_ewma_alpha=location_ewma_alpha,
        location_supply_capacity_per_tick=location_supply_capacity_per_tick,
        location_min_order_interval_ticks=location_min_order_interval_ticks,
        location_unfulfilled_unit_penalty=location_unfulfilled_unit_penalty,
        location_preferred_supplier_edge_id=location_preferred_supplier_edge_id,
        edge_lead_time_ticks=edge_lead_time_ticks,
        location_route_offset=location_route_offset,
        location_route_dst_location_id=location_route_dst_location_id,
        location_route_next_edge_id=location_route_next_edge_id,
        org_ext_id=org_ext_id,
        location_ext_id=location_ext_id,
        batch_ext_id=batch_ext_id,
        pack_ext_id=pack_ext_id,
    )
    engine_input.validate_shapes()
    return engine_input
