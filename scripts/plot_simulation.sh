SCENARIO=$"multi_market_sparse_scenario()"
NUM_TICKS=100

PYTHONPATH=python uv run python -c "
from compiler.compile import compile_scenario
from policy.scenarios_large import multi_market_sparse_scenario
from policy.scenarios import two_markets_demo
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import events_as_records, plot_event_timeline, plot_physical_locations

inp = compile_scenario($SCENARIO)
sim = create_native_simulator(inp)
sim.run_ticks($NUM_TICKS)
rec = events_as_records(sim, inp)
fig, ax = plot_event_timeline(rec)
fig.savefig('out_logs/timeline.png')
fig2, ax2 = plot_physical_locations(sim, inp)
fig2.savefig('out_logs/locations.png')
"