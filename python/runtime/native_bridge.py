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


def _scenario_summary(scenario: Scenario) -> str:
    return (
        f"seed={scenario.seed}, orgs={len(scenario.organizations)}, "
        f"locations={len(scenario.locations)}, products={len(scenario.products)}, "
        f"batches={len(scenario.batches)}, packs={len(scenario.packs)}, "
        f"edges={len(scenario.location_edges)}"
    )


def create_native_simulator(compiled: EngineInput, *, context: str | None = None):
    """Build a native Simulator from compiled EngineInput.

    If *context* is set, failures from the native layer are wrapped with that string
    and the original exception is chained (``raise ... from e``).
    """
    try:
        return _native.create_simulator(compiled)
    except Exception as e:
        if context is None:
            raise
        raise RuntimeError(f"Failed to create native simulator ({context})") from e


def compile_and_create_native_simulator(scenario: Scenario):
    """Compile a Scenario and build a native Simulator."""
    summary = _scenario_summary(scenario)
    try:
        compiled = compile_scenario(scenario)
    except Exception as e:
        raise RuntimeError(f"Failed to compile scenario ({summary})") from e
    return create_native_simulator(compiled, context=summary)

