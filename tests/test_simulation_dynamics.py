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
from policy.models import PackState
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
    sim = create_native_simulator(inp)
    sim.run_ticks(800)
    markets = set(sim.physical_pack_market_ids())
    assert len(markets) >= 2
    assert all(0 <= m < inp.n_markets for m in markets)
