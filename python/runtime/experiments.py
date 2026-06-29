"""End-to-end Bulgaria experiment helpers: build -> compile -> calibrate -> run.

These tie the canonical pipeline to the native engine and apply the realism calibration
the raw compiler defaults do not (the compiler is scenario-agnostic). Keeping calibration
here -- as an explicit transform on :class:`~compiler.types.EngineInput` -- means the
fraud/perturbation layer operates on the exact same calibrated engine parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from canonical.models import CanonicalDataset
from canonical.pipeline import build_bulgaria_scenario
from compiler.compile import compile_scenario
from compiler.enums import org_type_u8
from compiler.types import EngineInput
from policy.models import OrgType, Scenario

from .native_bridge import create_native_simulator
from .simulation_viz import events_as_records

OBP_U8 = int(org_type_u8(OrgType.OBP))
WHOLESALER_U8 = int(org_type_u8(OrgType.WHOLESALER))
LOCAL_ORG_U8 = int(org_type_u8(OrgType.LOCAL_ORG))


def location_org_types(inp: EngineInput) -> np.ndarray:
    """uint8 OrgType per location (resolved through ``location_org_id``)."""
    return np.asarray(inp.org_type)[np.asarray(inp.location_org_id)]


def calibrate_supply_policies(
    inp: EngineInput,
    *,
    obp_capacity_per_tick: int = 4,
    wholesaler_capacity_per_tick: int = 3,
) -> EngineInput:
    """Switch supply nodes to the push policy (id 2) in place.

    The default order-up-to policy gates a wholesaler's outflow on its own inventory
    position, so once stock accumulates it stops dispensing and packs pile up. The push
    policy ships ``capacity`` packs per tick along the preferred downstream edge every
    tick, producing continuous OBP -> wholesaler -> pharmacy flow. Returns ``inp``.
    """
    org_t = location_org_types(inp)
    for i in range(inp.n_locations):
        t = int(org_t[i])
        if t == OBP_U8:
            inp.location_supply_policy_id[i] = np.uint8(2)
            inp.location_supply_capacity_per_tick[i] = np.uint32(obp_capacity_per_tick)
        elif t == WHOLESALER_U8:
            inp.location_supply_policy_id[i] = np.uint8(2)
            inp.location_supply_capacity_per_tick[i] = np.uint32(
                wholesaler_capacity_per_tick
            )
    return inp


@dataclass
class BulgariaSimulationInput:
    scenario: Scenario
    canonical: CanonicalDataset
    engine_input: EngineInput


def build_bulgaria_simulation_input(
    *,
    bg_registry_path: str | Path = "data/data.json",
    spor_locations_path: str | Path = "data/spor_locations.csv",
    geocode: bool = True,
    geocode_cache_path: str | Path | None = "data/geocode_cache_bg.json",
    seed: int = 42,
    max_locations: int = 350,
    packs_per_obp: int = 60,
    packs_per_wholesaler: int = 12,
    n_obp_hubs: int = 6,
    calibrate: bool = True,
) -> BulgariaSimulationInput:
    """Build a calibrated, compile-ready Bulgaria engine input."""
    scenario, canonical = build_bulgaria_scenario(
        bg_registry_path=bg_registry_path,
        spor_locations_path=spor_locations_path,
        geocode=geocode,
        geocode_cache_path=geocode_cache_path,
        seed=seed,
        max_locations=max_locations,
        packs_per_obp=packs_per_obp,
        packs_per_wholesaler=packs_per_wholesaler,
        n_obp_hubs=n_obp_hubs,
    )
    inp = compile_scenario(scenario)
    if calibrate:
        calibrate_supply_policies(inp)
        inp.validate_shapes()
    return BulgariaSimulationInput(
        scenario=scenario, canonical=canonical, engine_input=inp
    )


def run_simulation(inp: EngineInput, ticks: int) -> tuple[object, list[dict]]:
    """Create a native simulator, run ``ticks``, and return ``(sim, event_records)``."""
    sim = create_native_simulator(inp, context="bulgaria_experiment")
    sim.run_ticks(ticks)
    return sim, events_as_records(sim, inp)
