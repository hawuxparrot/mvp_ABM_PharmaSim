SCENARIO=$"multi_market_sparse_scenario()"
NUM_TICKS_DEBUG=5

PYTHONPATH=python uv run python -c "
from compiler.compile import compile_scenario
from policy.scenarios_large import multi_market_sparse_scenario
from policy.scenarios import two_markets_demo
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import run_ticks_with_hook, print_tick_debug

inp = compile_scenario($SCENARIO)
sim = create_native_simulator(inp)
run_ticks_with_hook(sim, inp, $NUM_TICKS_DEBUG, print_tick_debug)
"