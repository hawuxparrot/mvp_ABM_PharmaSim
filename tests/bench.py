import pytest

from compiler.compile import compile_scenario_with_precomputed_edges
from policy.scenarios_large import multi_market_sparse_scenario_precomputed
from runtime.native_bridge import create_native_simulator

@pytest.fixture(scope="module")
def compiled():
    s_l, es, ed, ec, ek = multi_market_sparse_scenario_precomputed()
    return compile_scenario_with_precomputed_edges(s_l, es, ed, ec, ek)

def test_bench_create_simulator(benchmark,compiled):
    benchmark(create_native_simulator, compiled)

def test_bench_run_ticks(benchmark,compiled):
    sim = create_native_simulator(compiled)
    
    def run():
        sim.run_ticks(100_000)

    benchmark(run)