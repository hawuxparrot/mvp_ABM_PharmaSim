"""Small compile checks for policy.scenarios_large (full 10k×4 is not run in CI)."""

from __future__ import annotations

import pytest

from compiler.compile import compile_scenario
from policy.scenarios_large import multi_market_sparse_scenario


def test_multi_market_sparse_compiles_out_degree() -> None:
    s = multi_market_sparse_scenario(
        market_codes=("DE", "FR"),
        locations_per_market=80,
        seed=1,
        graph_model="out_degree",
        out_degree=5,
        wholesaler_site_count_per_market=8,
        local_org_pool_per_market=15,
        packs_per_market=5,
        behavior_location_fraction=0.1,
    )
    inp = compile_scenario(s)
    assert inp.n_locations == 160
    assert inp.n_markets == 2
    assert inp.n_edges > 0
    assert inp.n_packs == 10


def test_erdos_renyi_cap_warns() -> None:
    s = multi_market_sparse_scenario(
        market_codes=("DE",),
        locations_per_market=200,
        seed=2,
        graph_model="erdos_renyi",
        er_edge_probability=0.5,
        max_directed_edges_per_market=500,
    )
    inp = compile_scenario(s)
    assert inp.n_edges <= 500


def test_locations_per_market_too_small_raises() -> None:
    with pytest.raises(ValueError):
        multi_market_sparse_scenario(locations_per_market=2)
