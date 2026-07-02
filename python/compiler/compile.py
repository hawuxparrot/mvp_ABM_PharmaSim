"""Compile a validated :class:`~policy.models.Scenario` into :class:`~compiler.types.EngineInput`."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from canonical.substitutability import SubstitutabilityIndex
from compiler.enums import decommission_reason_u8, org_type_u8, pack_state_u8
from compiler.types import ENGINE_INPUT_SCHEMA_VERSION, EngineInput
from compiler.validate import validate_scenario
from policy.models import DecommissionReason, OrgType, Scenario

# ATC cutoff for substitution-on-stockout: level 4 (chemical subgroup) approximates clinical
# interchangeability (e.g. all statins in C10AA), which is the granularity the engine consumes.
_SUBSTITUTION_ATC_LEVEL = 4

# EMVS reactivation window: a decommissioned pack can be reverted to Active within this many
# ticks (FMD allows 10 days). Treating one tick ~ one day.
_REACTIVATION_WINDOW_TICKS = 10


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
    edge_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-source first-hop routing CSR via C-level all-sources Dijkstra.

    For every location ``s`` and reachable destination ``t`` we emit the id of the first
    edge on a shortest path ``s -> t`` (multihop is resolved hop-by-hop by the engine via
    ``lookup_next_edge``). Destinations are emitted in ascending id order per source so the
    layout matches the previous pure-Python implementation.

    Performance: shortest paths run inside ``scipy.sparse.csgraph`` (compiled), and we only
    run them from nodes that actually have outgoing edges (supply hubs/wholesalers). Pure
    sinks (pharmacies/hospitals) get empty route ranges, cutting the source count from
    ``n_loc`` to ``n_supply``.
    """
    off = np.zeros(n_loc + 1, dtype=np.uint32)
    empty = np.array([], dtype=np.uint32)
    if n_edge == 0 or n_loc == 0:
        return off, empty.copy(), empty.copy()

    src = np.asarray(edge_src_location_id, dtype=np.int64)
    dst = np.asarray(edge_dst_location_id, dtype=np.int64)
    cost = np.asarray(edge_cost, dtype=np.float64)

    # Dijkstra needs strictly positive weights to relax correctly; clamp zero/negative.
    weights = np.where(cost > 0.0, cost, 1e-6)
    graph = csr_matrix((weights, (src, dst)), shape=(n_loc, n_loc))

    # First-hop edge id for each (src, immediate-neighbor) pair, choosing the cheapest
    # parallel edge so the recovered hop matches the weight Dijkstra used.
    first_edge_of: dict[tuple[int, int], tuple[int, float]] = {}
    for eid in range(n_edge):
        key = (int(src[eid]), int(dst[eid]))
        c = float(cost[eid])
        cur = first_edge_of.get(key)
        if cur is None or c < cur[1]:
            first_edge_of[key] = (eid, c)

    sources = np.unique(src)
    dist, pred = dijkstra(
        graph, directed=True, indices=sources, return_predecessors=True
    )
    src_row = {int(s): i for i, s in enumerate(sources)}

    flat_dst: list[int] = []
    flat_next: list[int] = []
    for s in range(n_loc):
        row = src_row.get(s)
        if row is not None:
            d_row = dist[row]
            p_row = pred[row]
            # Resolve first hop for every reachable node, processing in increasing
            # distance so first_hop[pred[t]] is already known (memoized shortest-path tree).
            order = np.argsort(d_row, kind="stable")
            first_hop = np.full(n_loc, -1, dtype=np.int64)
            for t in order:
                t = int(t)
                if t == s or not np.isfinite(d_row[t]):
                    continue
                p = int(p_row[t])
                if p == s:
                    first_hop[t] = t
                elif p >= 0 and first_hop[p] >= 0:
                    first_hop[t] = first_hop[p]
                else:
                    # Ordering gap (e.g. equal-distance ties): recover by walking back.
                    cur = t
                    fh = -1
                    while True:
                        pc = int(p_row[cur])
                        if pc == s:
                            fh = cur
                            break
                        if pc < 0:
                            break
                        cur = pc
                    if fh < 0:
                        continue
                    first_hop[t] = fh
            # Emit in ascending destination id for a stable, matching CSR layout.
            for t in range(n_loc):
                fh = int(first_hop[t])
                if fh < 0:
                    continue
                edge = first_edge_of.get((s, fh))
                if edge is None:
                    continue
                flat_dst.append(t)
                flat_next.append(edge[0])
        off[s + 1] = len(flat_dst)

    return (
        off,
        np.array(flat_dst, dtype=np.uint32),
        np.array(flat_next, dtype=np.uint32),
    )


def _empty_edge_columns() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.uint32),
    )


def _edge_columns_from_location_edges(
    s: Scenario,
    loc_ext_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_edge = len(s.location_edges)
    if n_edge == 0:
        return _empty_edge_columns()
    edge_src_location_id = np.zeros(n_edge, dtype=np.uint32)
    edge_dst_location_id = np.zeros(n_edge, dtype=np.uint32)
    edge_cost = np.zeros(n_edge, dtype=np.float32)
    edge_capacity = np.zeros(n_edge, dtype=np.uint32)
    for i, edge in enumerate(s.location_edges):
        edge_src_location_id[i] = loc_ext_to_id[edge.src_location_ext_id]
        edge_dst_location_id[i] = loc_ext_to_id[edge.dst_location_ext_id]
        edge_cost[i] = np.float32(edge.cost)
        edge_capacity[i] = np.uint32(edge.capacity)
    return edge_src_location_id, edge_dst_location_id, edge_cost, edge_capacity


def _coerce_edge_columns(
    *,
    n_loc: int,
    edge_src_location_id: np.ndarray,
    edge_dst_location_id: np.ndarray,
    edge_cost: np.ndarray,
    edge_capacity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(edge_src_location_id, dtype=np.uint32)
    dst = np.asarray(edge_dst_location_id, dtype=np.uint32)
    cost = np.asarray(edge_cost, dtype=np.float32)
    capacity = np.asarray(edge_capacity, dtype=np.uint32)

    if src.ndim != 1 or dst.ndim != 1 or cost.ndim != 1 or capacity.ndim != 1:
        raise ValueError("edge columns must be 1-D arrays")
    if not (len(src) == len(dst) == len(cost) == len(capacity)):
        raise ValueError("edge columns must have equal length")
    if len(src) > 0:
        if int(src.max()) >= n_loc or int(dst.max()) >= n_loc:
            raise ValueError("edge endpoint references out-of-range location id")
        if np.any(src == dst):
            raise ValueError("edges may not contain self-loops")
        if np.any(cost < 0):
            raise ValueError("edge_cost must be non-negative")
        if np.any(capacity < 0):
            raise ValueError("edge_capacity must be non-negative")
    if not np.all(np.isfinite(cost)):
        raise ValueError("edge_cost must be finite")
    return src, dst, cost, capacity


def _resolve_edge_columns(
    s: Scenario,
    loc_ext_to_id: dict[str, int],
    *,
    edge_src_location_id: np.ndarray | None = None,
    edge_dst_location_id: np.ndarray | None = None,
    edge_cost: np.ndarray | None = None,
    edge_capacity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_args = (
        edge_src_location_id,
        edge_dst_location_id,
        edge_cost,
        edge_capacity,
    )
    if any(arg is not None for arg in edge_args):
        if not all(arg is not None for arg in edge_args):
            raise ValueError(
                "edge_src_location_id, edge_dst_location_id, edge_cost, and edge_capacity "
                "must all be provided together"
            )
        if len(s.location_edges) > 0:
            raise ValueError(
                "Scenario.location_edges must be empty when supplying edge columns "
                "to avoid ambiguous graph sources"
            )
        assert edge_src_location_id is not None
        assert edge_dst_location_id is not None
        assert edge_cost is not None
        assert edge_capacity is not None
        return _coerce_edge_columns(
            n_loc=len(s.locations),
            edge_src_location_id=edge_src_location_id,
            edge_dst_location_id=edge_dst_location_id,
            edge_cost=edge_cost,
            edge_capacity=edge_capacity,
        )
    return _edge_columns_from_location_edges(s, loc_ext_to_id)


def _compile_validated(
    s: Scenario,
    *,
    edge_columns: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> EngineInput:

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

    edge_src_location_id, edge_dst_location_id, edge_cost, edge_capacity = edge_columns
    n_edge = len(edge_src_location_id)

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

    # EMVS serial identity: intern serial strings to dense ids. Distinct packs sharing a serial
    # collapse to the same id, which is how the engine detects clones (DUPLICATE_SERIAL).
    serial_to_id: dict[str, int] = {}
    pack_serial_id = np.zeros(n_pack, dtype=np.uint32)
    for i, serial in enumerate(pack_serial):
        sid = serial_to_id.get(serial)
        if sid is None:
            sid = len(serial_to_id)
            serial_to_id[serial] = sid
        pack_serial_id[i] = np.uint32(sid)
    n_serials = len(serial_to_id)

    location_has_behavior = np.zeros(n_loc, dtype=np.uint8)
    location_verify_prob = np.zeros(n_loc, dtype=np.float32)
    location_decommission_prob = np.zeros(n_loc, dtype=np.float32)
    location_reactivate_prob = np.zeros(n_loc, dtype=np.float32)
    # Default decommission reason is DISPENSED (normal point-of-dispense outcome); fraud drivers
    # override specific sites (e.g. STOLEN at a distributor) post-compile.
    location_decommission_reason_id = np.full(
        n_loc, decommission_reason_u8(DecommissionReason.DISPENSED), dtype=np.uint8
    )
    location_demand_policy_id = np.zeros(n_loc, dtype=np.uint8)
    location_supply_policy_id = np.zeros(n_loc, dtype=np.uint8)
    location_initial_on_hand = np.zeros(n_loc, dtype=np.int32)
    location_initial_backlog = np.zeros(n_loc, dtype=np.int32)
    location_initial_pipeline_outstanding = np.zeros(n_loc, dtype=np.int32)
    location_demand_const_rate = np.zeros(n_loc, dtype=np.int32)
    location_demand_poisson_lambda = np.zeros(n_loc, dtype=np.float32)
    location_reorder_point_s = np.zeros(n_loc, dtype=np.int32)
    location_order_up_to_S = np.zeros(n_loc, dtype=np.int32)
    location_unfulfilled_unit_penalty = np.zeros(n_loc, dtype=np.float32)
    location_preferred_supplier_edge_id = np.full(n_loc, np.uint32(2**32 - 1), dtype=np.uint32)
    location_base_stock_level = np.zeros(n_loc, dtype=np.int32)
    location_ewma_alpha = np.zeros(n_loc, dtype=np.float32)
    location_supply_capacity_per_tick = np.zeros(n_loc, dtype=np.uint32)
    location_min_order_interval_ticks = np.zeros(n_loc, dtype=np.uint32)
    location_penalty_policy_id = np.zeros(n_loc, dtype=np.uint8)
    edge_lead_time_ticks = np.zeros(n_edge, dtype=np.uint16)
    for i, loc in enumerate(s.locations):
        beh = s.behavior_by_location.get(loc.ext_id)
        if beh is None:
            continue
        location_has_behavior[i] = 1
        location_verify_prob[i] = np.float32(beh.verify_prob)
        location_decommission_prob[i] = np.float32(beh.decomission_prob)
        location_reactivate_prob[i] = np.float32(beh.reactivate_prob)

    OBP_U8 = int(org_type_u8(OrgType.OBP))
    LOCAL_ORG_U8 = int(org_type_u8(OrgType.LOCAL_ORG))
    WHOLESALER_U8 = int(org_type_u8(OrgType.WHOLESALER))
    for i in range(n_loc):
        org_row = int(location_org_id[i])
        ot = int(org_type[org_row])
        if ot == LOCAL_ORG_U8:
            location_demand_policy_id[i] = np.uint8(2)
            location_demand_poisson_lambda[i] = np.float32(1.0)
            location_penalty_policy_id[i] = np.uint8(1)
            location_unfulfilled_unit_penalty[i] = np.float32(0.1)
        elif ot == WHOLESALER_U8:
            location_supply_policy_id[i] = np.uint8(1)
            location_supply_capacity_per_tick[i] = np.uint32(10)
            location_order_up_to_S[i] = np.int32(20)
            start = int(location_out_edge_offset[i])
            end = int(location_out_edge_offset[i + 1])
            if start < end:
                location_preferred_supplier_edge_id[i] = np.uint32(int(location_out_edge_id[start]))
        elif ot == OBP_U8:
            location_supply_policy_id[i] = np.uint8(2)
            location_supply_capacity_per_tick[i] = np.uint32(5)
            start = int(location_out_edge_offset[i])
            end = int(location_out_edge_offset[i + 1])
            if start < end:
                location_preferred_supplier_edge_id[i] = np.uint32(int(location_out_edge_id[start]))

    # products: uniform demand mix by default (callers may reweight post-compile, e.g.
    # runtime.experiments.apply_product_demand_weights), plus an ATC-derived substitute CSR
    # consumed by the engine's substitution-on-stockout path.
    product_demand_weight = np.ones(n_prod, dtype=np.float32)
    sub_index = SubstitutabilityIndex(s.products, default_level=_SUBSTITUTION_ATC_LEVEL)
    sub_offsets: list[int] = [0]
    sub_flat: list[int] = []
    for p in s.products:
        for sid in sub_index.substitutes(p.ext_id, level=_SUBSTITUTION_ATC_LEVEL):
            sub_flat.append(product_ext_to_id[sid])
        sub_offsets.append(len(sub_flat))
    product_substitute_offset = np.array(sub_offsets, dtype=np.uint32)
    product_substitute_id = np.array(sub_flat, dtype=np.uint32)

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
        n_serials=n_serials,
        reactivation_window_ticks=_REACTIVATION_WINDOW_TICKS,
        market_code=market_code,
        product_demand_weight=product_demand_weight,
        product_substitute_offset=product_substitute_offset,
        product_substitute_id=product_substitute_id,
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
        pack_serial_id=pack_serial_id,
        location_has_behavior=location_has_behavior,
        location_verify_prob=location_verify_prob,
        location_decommission_prob=location_decommission_prob,
        location_reactivate_prob=location_reactivate_prob,
        location_decommission_reason_id=location_decommission_reason_id,
        location_demand_policy_id=location_demand_policy_id,
        location_supply_policy_id=location_supply_policy_id,
        location_initial_on_hand=location_initial_on_hand,
        location_initial_backlog=location_initial_backlog,
        location_initial_pipeline_outstanding=location_initial_pipeline_outstanding,
        location_demand_const_rate=location_demand_const_rate,
        location_demand_poisson_lambda=location_demand_poisson_lambda,
        location_reorder_point_s=location_reorder_point_s,
        location_order_up_to_S=location_order_up_to_S,
        location_base_stock_level=location_base_stock_level,
        location_ewma_alpha=location_ewma_alpha,
        location_supply_capacity_per_tick=location_supply_capacity_per_tick,
        location_min_order_interval_ticks=location_min_order_interval_ticks,
        location_penalty_policy_id=location_penalty_policy_id,
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


def compile_scenario(
    scenario: Scenario,
    *,
    edge_src_location_id: np.ndarray | None = None,
    edge_dst_location_id: np.ndarray | None = None,
    edge_cost: np.ndarray | None = None,
    edge_capacity: np.ndarray | None = None,
) -> EngineInput:
    """
    Validate *scenario*, assign dense IDs (list order = row id), build columnar :class:`EngineInput`.

    Edge graph ingress is always columnar internally. By default, edges are taken from
    ``scenario.location_edges`` and converted to dense columns. Large builders may pass
    pre-built edge columns instead (``Scenario.location_edges`` must then be empty).

    ID policy: for each entity list on :class:`~policy.models.Scenario`, ``row_index`` is the dense
    id (0 .. n-1). Markets are interned: sorted unique market strings → ``market_id``.

    Location behavior (Option B): always emit dense arrays of length ``n_locations``; rows are
    zero unless :attr:`~policy.models.Scenario.behavior_by_location` defines a row for that site.
    """
    s = validate_scenario(scenario)
    loc_ext_to_id = {loc.ext_id: i for i, loc in enumerate(s.locations)}
    edge_columns = _resolve_edge_columns(
        s,
        loc_ext_to_id,
        edge_src_location_id=edge_src_location_id,
        edge_dst_location_id=edge_dst_location_id,
        edge_cost=edge_cost,
        edge_capacity=edge_capacity,
    )
    return _compile_validated(s, edge_columns=edge_columns)
