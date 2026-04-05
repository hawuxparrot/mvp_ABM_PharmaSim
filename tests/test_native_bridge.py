"""End-to-end checks for the compiled scenario → native simulator path."""

from __future__ import annotations

import importlib.util

import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from policy.scenarios import two_markets_demo
from runtime.native_bridge import compile_and_create_native_simulator


def test_native_simulator_two_markets_run_ticks_advances_clock() -> None:
    sim = compile_and_create_native_simulator(two_markets_demo())
    sim.run_ticks(3)
    assert sim.current_tick() == 3
