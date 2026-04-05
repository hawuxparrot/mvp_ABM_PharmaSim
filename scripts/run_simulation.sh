SCENARIO=$"multi_market_sparse_scenario()"
NUM_TICKS=5000

#PYTHONPATH=python uv run python -c "
# from compiler.compile import compile_scenario
# from policy.scenarios import two_markets_demo
# from runtime.native_bridge import create_native_simulator
# from runtime.simulation_viz import export_run_report
# inp = compile_scenario($SCENARIO)
# sim = create_native_simulator(inp)
# sim.run_ticks($NUM_TICKS)
# print('tick', sim.current_tick(), 'events', sim.event_count())
# export_run_report(sim, inp, 'out_logs', prefix='run')
# "

PYTHONPATH=python uv run python -c "
from compiler.compile import compile_scenario
from policy.scenarios_large import multi_market_sparse_scenario
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import export_run_report
inp = compile_scenario($SCENARIO)
sim = create_native_simulator(inp)
sim.run_ticks($NUM_TICKS)
print('tick', sim.current_tick(), 'events', sim.event_count())
export_run_report(sim, inp, 'out_logs', prefix='run')
"