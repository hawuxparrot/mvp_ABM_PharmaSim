"""Tests for runtime.simulation_viz (no matplotlib required)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from compiler.compile import compile_scenario
from policy.scenarios import two_markets_demo
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import (
    dump_debug_report_to_string,
    events_as_records,
    export_run_report,
    format_pack_history_text,
    physical_snapshot_rows,
    run_ticks_with_hook,
    write_events_csv,
)


def test_events_as_records_aligns_with_simulator() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(15)
    rec = events_as_records(sim, inp)
    assert len(rec) == sim.event_count()
    if rec:
        assert "tick" in rec[0] and "event_type" in rec[0] and "pack_ext_id" in rec[0]


def test_export_run_report_writes_files(tmp_path: Path) -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(5)
    export_run_report(sim, inp, tmp_path, prefix="t")
    csv_path = tmp_path / "t_events.csv"
    assert csv_path.is_file()
    js = json.loads((tmp_path / "t_snapshot_final.json").read_text(encoding="utf-8"))
    assert len(js) == inp.n_packs


def test_run_ticks_with_hook_called_each_tick() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    ticks_seen: list[int] = []

    def hook(info) -> None:
        ticks_seen.append(info.simulator_tick)

    run_ticks_with_hook(sim, inp, 4, hook)
    assert ticks_seen == [1, 2, 3, 4]


def test_dump_debug_report_non_empty() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(3)
    text = dump_debug_report_to_string(sim, inp)
    assert "current_tick=3" in text
    assert "physical snapshot" in text


def test_write_events_csv_roundtrip_header(tmp_path: Path) -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(2)
    rec = events_as_records(sim, inp)
    p = tmp_path / "e.csv"
    write_events_csv(rec, p)
    first = p.read_text(encoding="utf-8").splitlines()[0]
    assert "tick" in first and "event_type" in first


def test_pack_history_contains_moves() -> None:
    inp = compile_scenario(two_markets_demo())
    sim = create_native_simulator(inp)
    sim.run_ticks(100)
    rec = events_as_records(sim, inp)
    text = format_pack_history_text(rec, inp, 0)
    assert "pack_" in text or "pack#" in text
    assert "location trace" in text
