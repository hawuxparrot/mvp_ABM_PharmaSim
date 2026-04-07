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

PYTHONPATH=python uv run python <<SCRIPT
from compiler.compile import compile_scenario
from policy.scenarios_large import multi_market_sparse_scenario
from policy.scenarios import two_markets_demo
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import (
    export_run_report,
    events_as_records,
    physical_snapshot_rows,
    plot_event_timeline,
    plot_physical_locations,
    pack_move_trace,
    format_pack_history_text,
    dump_debug_report_to_string,
    format_events_text,
    format_snapshot_text,
    filter_events_for_pack,
)

inp = compile_scenario($SCENARIO)
sim = create_native_simulator(inp)
sim.run_ticks($NUM_TICKS)

rec = events_as_records(sim, inp)
print(format_events_text(rec, max_rows=10))
print(format_snapshot_text(physical_snapshot_rows(sim, inp)))
print(format_pack_history_text(rec, inp, pack_id=0))
print(filter_events_for_pack(rec, 0)[:5])
export_run_report(sim, inp, 'out_logs', prefix='run')
SCRIPT