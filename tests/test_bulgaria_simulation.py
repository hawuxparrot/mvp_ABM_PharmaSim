"""End-to-end Bulgaria simulation: build -> compile -> calibrate -> run -> ground-truth log."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from runtime.experiments import (  # noqa: E402
    LOCAL_ORG_U8,
    build_bulgaria_simulation_input,
    location_org_types,
    run_simulation,
)
from runtime.simulation_viz import export_run_report  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _sim_input(max_locations: int = 250):
    return build_bulgaria_simulation_input(
        bg_registry_path=str(REPO / "data" / "data.json"),
        spor_locations_path=str(REPO / "data" / "spor_locations.csv"),
        geocode=False,
        max_locations=max_locations,
        packs_per_obp=40,
        packs_per_wholesaler=10,
        n_obp_hubs=6,
    )


def test_bulgaria_scenario_has_connected_graph() -> None:
    si = _sim_input()
    inp = si.engine_input
    assert inp.n_locations > 0
    # The connectivity-first builder must not collapse to an edgeless graph.
    assert inp.n_edges > 0
    assert inp.n_packs > 0


def test_bulgaria_simulation_moves_packs_to_terminals() -> None:
    si = _sim_input()
    inp = si.engine_input
    org_t = location_org_types(inp)
    sim, records = run_simulation(inp, 250)

    event_counts = Counter(r["event_type"] for r in records)
    assert event_counts.get("MOVE", 0) > 0

    moves = [r for r in records if r["event_type"] == "MOVE"]
    terminals_reached = {
        r["to_location_id"]
        for r in moves
        if r["to_location_id"] is not None
        and int(org_t[r["to_location_id"]]) == LOCAL_ORG_U8
    }
    # Packs must actually reach pharmacy/hospital terminals, across more than one site.
    assert len(terminals_reached) >= 2


def test_bulgaria_ground_truth_log_export(tmp_path) -> None:
    si = _sim_input()
    sim, records = run_simulation(si.engine_input, 150)
    export_run_report(sim, si.engine_input, tmp_path, prefix="bg")
    assert (tmp_path / "bg_events.csv").exists()
    assert (tmp_path / "bg_events.jsonl").exists()
    assert (tmp_path / "bg_snapshot_final.json").exists()
    assert (tmp_path / "bg_events.csv").stat().st_size > 0


def test_bulgaria_simulation_deterministic() -> None:
    si1 = _sim_input()
    _, rec1 = run_simulation(si1.engine_input, 120)
    si2 = _sim_input()
    _, rec2 = run_simulation(si2.engine_input, 120)
    assert len(rec1) == len(rec2)
    assert [r["event_type"] for r in rec1[:200]] == [r["event_type"] for r in rec2[:200]]
