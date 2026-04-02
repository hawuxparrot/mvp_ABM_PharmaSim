from __future__ import annotations

import importlib

from compiler.compile import compile_scenario
from compiler.types import EngineInput
from policy.models import Scenario

try:
    from . import _pharmasim_native as _native  # type: ignore[import-untyped]
except ImportError:
    # Local dev fallback when extension is built in-place but not installed into runtime package.
    _native = importlib.import_module("_pharmasim_native")


def create_native_simulator(compiled: EngineInput):
    """Build a native Simulator from compiled EngineInput."""
    return _native.create_simulator(compiled)


def compile_and_create_native_simulator(scenario: Scenario):
    """Compile a Scenario and build a native Simulator."""
    compiled = compile_scenario(scenario)
    return create_native_simulator(compiled)

