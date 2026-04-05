import pytest

from compiler.compile import compile_scenario
from policy.scenarios import two_markets_demo
from policy.scenarios_large import multi_market_sparse_scenario
from runtime.native_bridge import create_native_simulator

@pytest.fixture(scope="module")
def compiled():
    s = multi_market_sparse_scenario()
    # s = two_markets_demo()
    return compile_scenario(s)

def test_bench_create_simulator(benchmark,compiled):
    benchmark(create_native_simulator, compiled)

def test_bench_run_ticks(benchmark,compiled):
    sim = create_native_simulator(compiled)
    
    def run():
        sim.run_ticks(100_000)

    benchmark(run)