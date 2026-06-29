#!/usr/bin/env bash
set -euo pipefail

# End-to-end Bulgaria run: build canonical graph -> compile -> calibrate -> simulate ->
# export the ground-truth event log and a map/flow plot under out_logs/.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TICKS="${TICKS:-300}"
MAX_LOCATIONS="${MAX_LOCATIONS:-350}"

PYTHONPATH=python uv run python <<SCRIPT
from runtime.experiments import build_bulgaria_simulation_input, run_simulation
from runtime.simulation_viz import export_run_report, plot_location_graph
from collections import Counter

si = build_bulgaria_simulation_input(
    max_locations=${MAX_LOCATIONS},
    geocode=True,
    geocode_cache_path="data/geocode_cache_bg.json",
)
inp = si.engine_input
print(f"locations={inp.n_locations} edges={inp.n_edges} packs={inp.n_packs}")

sim, records = run_simulation(inp, ${TICKS})
print("events:", dict(Counter(r["event_type"] for r in records)))

export_run_report(sim, inp, "out_logs", prefix="bulgaria")
print("wrote out_logs/bulgaria_events.csv / .jsonl / _snapshot_final.json")

try:
    fig, ax = plot_location_graph(si.canonical, si.scenario, title="Bulgaria distribution graph")
    fig.savefig("out_logs/bulgaria_graph.png", dpi=130)
    print("wrote out_logs/bulgaria_graph.png")
except Exception as e:  # matplotlib optional
    print(f"graph plot skipped: {e}")
SCRIPT
