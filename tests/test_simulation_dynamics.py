"""Invariants and determinism for the native simulation kernel (MVP dynamics)."""

from __future__ import annotations

import importlib.util

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from compiler.compile import compile_scenario
from compiler.enums import PACK_STATE_TO_U8
from policy.models import (
    Batch,
    Location,
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
from policy.scenarios import two_markets_demo
from runtime.native_bridge import (
    compile_and_create_native_simulator,
    create_native_simulator,
)

DECOM_U8 = PACK_STATE_TO_U8[PackState.DECOMISSIONED]
MOVE_U8 = 3


def test_registry_mirror_matches_physical_many_ticks() -> None:
    sim = compile_and_create_native_simulator(two_markets_demo())
    sim.run_ticks(500)
    assert sim.registry_matches_physical()
    assert sim.current_tick() == 500


def test_pack_locations_stay_within_graph() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(150)
    n_loc = inp.n_locations
    for loc in sim.physical_pack_location_ids():
        assert 0 <= loc < n_loc


def test_deterministic_replay_same_seed() -> None:
    s1 = compile_and_create_native_simulator(two_markets_demo())
    s1.run_ticks(120)
    s2 = compile_and_create_native_simulator(two_markets_demo())
    s2.run_ticks(120)
    assert list(s1.physical_pack_states()) == list(s2.physical_pack_states())
    assert list(s1.physical_pack_location_ids()) == list(s2.physical_pack_location_ids())
    assert list(s1.physical_pack_market_ids()) == list(s2.physical_pack_market_ids())
    assert s1.event_count() == s2.event_count()
    assert list(s1.event_log_types()) == list(s2.event_log_types())


def test_decommissioned_packs_do_not_move_afterwards() -> None:
    sim = compile_and_create_native_simulator(two_markets_demo())
    sim.run_ticks(250)
    states = list(sim.physical_pack_states())
    locs_before = list(sim.physical_pack_location_ids())
    sim.run_ticks(80)
    locs_after = list(sim.physical_pack_location_ids())
    for i, st in enumerate(states):
        if st == DECOM_U8:
            assert locs_after[i] == locs_before[i]


def test_some_moves_occur_on_two_markets_demo() -> None:
    sim = compile_and_create_native_simulator(two_markets_demo())
    sim.run_ticks(200)
    types = list(sim.event_log_types())
    assert MOVE_U8 in types


def test_cross_market_pack_market_ids_possible() -> None:
    """Graph includes DE→FR path; long runs should place packs in more than one market column."""
    inp = compile_scenario(two_markets_demo())
    obp_loc = inp.location_ext_id.index("loc_obp_de")
    wh_fr_loc = inp.location_ext_id.index("loc_wh_fr")
    for edge_id in range(inp.n_edges):
        if (
            int(inp.edge_src_location_id[edge_id]) == obp_loc
            and int(inp.edge_dst_location_id[edge_id]) == wh_fr_loc
        ):
            inp.location_preferred_supplier_edge_id[obp_loc] = edge_id
            break
    sim = create_native_simulator(inp)
    sim.run_ticks(800)
    markets = set(sim.physical_pack_market_ids())
    fr_id = inp.market_code.index("FR")
    assert fr_id in markets
    assert all(0 <= m < inp.n_markets for m in markets)


def test_poisson_demand_is_deterministic_with_fixed_seed() -> None:
    s1 = compile_and_create_native_simulator(two_markets_demo())
    s1.run_ticks(120)
    s2 = compile_and_create_native_simulator(two_markets_demo())
    s2.run_ticks(120)
    assert list(s1.location_backlog()) == list(s2.location_backlog())
    assert list(s1.location_cum_unfulfilled_penalty()) == pytest.approx(
        list(s2.location_cum_unfulfilled_penalty())
    )


def test_unfulfilled_penalty_is_monotone_non_decreasing() -> None:
    sim = compile_and_create_native_simulator(two_markets_demo())
    prev = list(sim.location_cum_unfulfilled_penalty())
    for _ in range(8):
        sim.run_ticks(25)
        cur = list(sim.location_cum_unfulfilled_penalty())
        for i in range(len(cur)):
            assert cur[i] >= prev[i]
        prev = cur


def _capacity_limited_lane_scenario() -> Scenario:
    orgs = [
        Organization(ext_id="obp", org_type=OrgType.OBP),
        Organization(ext_id="wh", org_type=OrgType.WHOLESALER),
        Organization(ext_id="ph", org_type=OrgType.LOCAL_ORG),
    ]
    locs = [
        Location(ext_id="loc_obp", org_ext_id="obp", market_code="DE", postal_code="10000"),
        Location(ext_id="loc_wh", org_ext_id="wh", market_code="DE", postal_code="20000"),
        Location(ext_id="loc_ph", org_ext_id="ph", market_code="DE", postal_code="30000"),
    ]
    products = [
        Product(
            ext_id="prod",
            codes=(ProductCode(ProductCodeScheme.GTIN, "01234567890123", True),),
        )
    ]
    batches = [
        Batch(
            ext_id="batch",
            product_ext_id="prod",
            manufacturer_org_ext_id="obp",
            intended_markets=("DE",),
        )
    ]
    packs = [
        Pack(
            ext_id=f"pack_{i}",
            product_ext_id="prod",
            batch_ext_id="batch",
            serial=f"SN{i:06d}",
            initial_market_code="DE",
            initial_location_ext_id="loc_wh",
            initial_state=PackState.ACTIVE,
        )
        for i in range(20)
    ]
    edges = [
        LocationEdge(
            src_location_ext_id="loc_obp",
            dst_location_ext_id="loc_wh",
            cost=1.0,
            capacity=10,
        ),
        LocationEdge(
            src_location_ext_id="loc_wh",
            dst_location_ext_id="loc_ph",
            cost=1.0,
            capacity=1,
        ),
    ]
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location={},
        seed=7,
    )


def test_edge_capacity_is_enforced_per_tick_on_supply_lane() -> None:
    inp = compile_scenario(_capacity_limited_lane_scenario())
    wh_loc = inp.location_ext_id.index("loc_wh")
    ph_loc = inp.location_ext_id.index("loc_ph")
    inp.location_supply_capacity_per_tick[wh_loc] = 100
    inp.location_demand_policy_id[ph_loc] = 1
    inp.location_demand_const_rate[ph_loc] = 10
    inp.location_demand_poisson_lambda[ph_loc] = 0.0
    sim = create_native_simulator(inp)
    sim.run_ticks(8)
    ticks = list(sim.event_log_ticks())
    types = list(sim.event_log_types())
    from_locs = list(sim.event_log_from_locations())
    to_locs = list(sim.event_log_to_locations())
    moved_per_tick: dict[int, int] = {}
    for tick, ev_t, src, dst in zip(ticks, types, from_locs, to_locs):
        if ev_t != MOVE_U8:
            continue
        if src == wh_loc and dst == ph_loc:
            moved_per_tick[tick] = moved_per_tick.get(tick, 0) + 1
    assert moved_per_tick
    assert max(moved_per_tick.values()) <= 1


def test_obp_pool_activation_reduces_decommissioned_stock() -> None:
    inp = compile_scenario(two_markets_demo())
    obp_loc = inp.location_ext_id.index("loc_obp_de")
    inp.pack_initial_location_id[:] = obp_loc
    inp.pack_initial_state[:] = DECOM_U8
    sim = create_native_simulator(inp)
    before = list(sim.physical_pack_states()).count(DECOM_U8)
    sim.run_ticks(3)
    after = list(sim.physical_pack_states()).count(DECOM_U8)
    assert after < before


def test_penalty_uses_end_of_tick_backlog() -> None:
    inp = compile_scenario(two_markets_demo())
    inp.location_supply_policy_id[:] = 0
    inp.location_demand_policy_id[:] = 0
    inp.location_demand_const_rate[:] = 0
    inp.location_penalty_policy_id[:] = 0
    inp.location_unfulfilled_unit_penalty[:] = 0.0

    target = inp.location_ext_id.index("loc_ph_de")
    inp.location_demand_policy_id[target] = 1
    inp.location_demand_const_rate[target] = 1
    inp.location_penalty_policy_id[target] = 1
    inp.location_unfulfilled_unit_penalty[target] = 2.0

    sim = create_native_simulator(inp)
    sim.run_ticks(3)

    penalties = list(sim.location_cum_unfulfilled_penalty())
    backlogs = list(sim.location_backlog())
    assert backlogs[target] == 3
    assert penalties[target] == pytest.approx(12.0)


def test_wholesaler_policy_id1_uses_order_up_to_inventory_gap() -> None:
    inp = compile_scenario(_capacity_limited_lane_scenario())
    wh_loc = inp.location_ext_id.index("loc_wh")
    inp.location_initial_on_hand[wh_loc] = 25
    inp.location_order_up_to_S[wh_loc] = 20
    inp.location_supply_capacity_per_tick[wh_loc] = 100

    sim = create_native_simulator(inp)
    sim.run_ticks(5)

    ticks = list(sim.event_log_ticks())
    types = list(sim.event_log_types())
    from_locs = list(sim.event_log_from_locations())
    to_locs = list(sim.event_log_to_locations())
    ph_loc = inp.location_ext_id.index("loc_ph")
    wh_moves = [
        (tick, src, dst)
        for tick, ev_t, src, dst in zip(ticks, types, from_locs, to_locs)
        if ev_t == MOVE_U8 and src == wh_loc and dst == ph_loc
    ]
    assert not wh_moves
