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
from runtime.simulation_viz import events_as_records

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
    """Graph includes a DE OBP -> FR wholesaler lane; multi-edge fan-out should naturally
    push stock down it (no manual preferred-edge steering), placing packs in the FR market."""
    inp = compile_scenario(two_markets_demo())
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


def _two_product_substitution_scenario() -> Scenario:
    """Wholesaler stocked only with product B; pharmacy demands product A (an ATC substitute).

    A is atorvastatin (C10AA05), B is simvastatin (C10AA01): same ATC4 subgroup (C10AA), so the
    engine may use arriving B packs to satisfy unmet A demand (substitution-on-stockout).
    """
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
            ext_id="A",
            codes=(ProductCode(ProductCodeScheme.GTIN, "00000000000001", True),),
            atc_code="C10AA05",
        ),
        Product(
            ext_id="B",
            codes=(ProductCode(ProductCodeScheme.GTIN, "00000000000002", True),),
            atc_code="C10AA01",
        ),
    ]
    batches = [
        Batch(ext_id="batchB", product_ext_id="B", manufacturer_org_ext_id="obp", intended_markets=("DE",)),
    ]
    packs = [
        Pack(
            ext_id=f"pack_{i}",
            product_ext_id="B",
            batch_ext_id="batchB",
            serial=f"SN{i:06d}",
            initial_market_code="DE",
            initial_location_ext_id="loc_wh",
            initial_state=PackState.ACTIVE,
        )
        for i in range(40)
    ]
    edges = [
        LocationEdge("loc_obp", "loc_wh", cost=1.0, capacity=50),
        LocationEdge("loc_wh", "loc_ph", cost=1.0, capacity=50),
    ]
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location={},
        seed=11,
    )


def _configure_substitution_demand(inp) -> tuple[int, int]:
    """Force constant product-A demand at the pharmacy and a push wholesaler. Returns (ph, A)."""
    n_loc = inp.n_locations
    for i in range(n_loc):
        inp.location_demand_policy_id[i] = 0
        inp.location_supply_policy_id[i] = 0
        inp.location_penalty_policy_id[i] = 0
    ph = inp.location_ext_id.index("loc_ph")
    wh = inp.location_ext_id.index("loc_wh")
    # All demand goes to product A (index 0); pharmacy accrues a steady A backlog.
    inp.product_demand_weight[:] = 0.0
    inp.product_demand_weight[0] = 1.0
    inp.location_demand_policy_id[ph] = 1
    inp.location_demand_const_rate[ph] = 2
    inp.location_demand_poisson_lambda[ph] = 0.0
    # Wholesaler pushes its (product-B) stock downstream to the pharmacy.
    inp.location_supply_policy_id[wh] = 2
    inp.location_supply_capacity_per_tick[wh] = 50
    return ph, 0


def _fanout_hub_scenario() -> Scenario:
    """One OBP hub feeding three wholesalers; tests that push fan-out spreads flow."""
    orgs = [Organization(ext_id="obp", org_type=OrgType.OBP)]
    locs = [Location(ext_id="loc_obp", org_ext_id="obp", market_code="DE", postal_code="10000")]
    for i in range(3):
        orgs.append(Organization(ext_id=f"wh{i}", org_type=OrgType.WHOLESALER))
        locs.append(
            Location(ext_id=f"loc_wh{i}", org_ext_id=f"wh{i}", market_code="DE", postal_code=f"2000{i}")
        )
    products = [
        Product(ext_id="prod", codes=(ProductCode(ProductCodeScheme.GTIN, "01234567890123", True),))
    ]
    batches = [Batch(ext_id="b", product_ext_id="prod", manufacturer_org_ext_id="obp", intended_markets=("DE",))]
    packs = [
        Pack(
            ext_id=f"pack_{i}",
            product_ext_id="prod",
            batch_ext_id="b",
            serial=f"SN{i:06d}",
            initial_market_code="DE",
            initial_location_ext_id="loc_obp",
            initial_state=PackState.ACTIVE,
        )
        for i in range(60)
    ]
    edges = [
        LocationEdge("loc_obp", f"loc_wh{i}", cost=1.0, capacity=20) for i in range(3)
    ]
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location={},
        seed=5,
    )


def test_push_fanout_spreads_across_downstream_neighbors() -> None:
    inp = compile_scenario(_fanout_hub_scenario())
    obp = inp.location_ext_id.index("loc_obp")
    inp.location_supply_policy_id[obp] = 2
    inp.location_supply_capacity_per_tick[obp] = 6
    sim = create_native_simulator(inp)
    sim.run_ticks(10)
    dests = {
        r["to_location_id"]
        for r in events_as_records(sim, inp)
        if r["event_type"] == "MOVE" and r["to_location_id"] is not None
    }
    wh_ids = {inp.location_ext_id.index(f"loc_wh{i}") for i in range(3)}
    # Fan-out must reach all three wholesalers, not funnel into a single preferred lane.
    assert wh_ids <= dests


def test_v7_per_product_state_shapes() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    assert sim.n_products() == inp.n_products
    by_prod = list(sim.location_backlog_by_product())
    assert len(by_prod) == inp.n_locations * inp.n_products
    # Aggregate accessor sums the per-product layout.
    agg = list(sim.location_backlog())
    assert len(agg) == inp.n_locations


def test_v7_substitution_on_stockout_reduces_backlog() -> None:
    inp = compile_scenario(_two_product_substitution_scenario())
    ph, prod_a = _configure_substitution_demand(inp)
    sim = create_native_simulator(inp)
    sim.run_ticks(40)
    n_prod = inp.n_products
    backlog_a_with_sub = list(sim.location_backlog_by_product())[ph * n_prod + prod_a]

    # Control: disable substitution (empty substitute CSR). Now arriving B packs cannot
    # satisfy A demand, so A's backlog must be strictly larger.
    inp2 = compile_scenario(_two_product_substitution_scenario())
    _configure_substitution_demand(inp2)
    import numpy as np

    inp2.product_substitute_offset = np.zeros(inp2.n_products + 1, dtype=np.uint32)
    inp2.product_substitute_id = np.array([], dtype=np.uint32)
    sim2 = create_native_simulator(inp2)
    sim2.run_ticks(40)
    backlog_a_no_sub = list(sim2.location_backlog_by_product())[ph * n_prod + prod_a]

    assert backlog_a_no_sub > backlog_a_with_sub


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
