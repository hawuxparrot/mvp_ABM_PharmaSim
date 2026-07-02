"""EMVS/NMVS kernel semantics: typed alerts, registry divergence, reactivation window.

These are tight unit tests over the native engine. They build a minimal scenario, force the
verify/decommission/reactivate probabilities to deterministic values, and freeze the supply
policies so pack movement does not perturb the assertions -- isolating the EMVS transaction
logic added on top of the supply-chain kernel.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if (
    importlib.util.find_spec("runtime._pharmasim_native") is None
    and importlib.util.find_spec("_pharmasim_native") is None
):
    pytest.skip("native extension _pharmasim_native not available", allow_module_level=True)

from compiler.compile import compile_scenario
from policy.models import (
    Batch,
    Location,
    LocationBehavior,
    LocationEdge,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario,
)
from runtime.native_bridge import create_native_simulator
from runtime.simulation_viz import events_as_records


def _product() -> Product:
    return Product(
        ext_id="prod",
        codes=(ProductCode(ProductCodeScheme.GTIN, "00000000000001", True),),
        atc_code="C10AA05",
    )


def _scenario(packs: list[Pack], *, market: str = "BG", behavior_verify: float = 0.0,
              behavior_decom: float = 0.0, behavior_react: float = 0.0) -> Scenario:
    orgs = [
        Organization(ext_id="obp", org_type=OrgType.OBP),
        Organization(ext_id="wh", org_type=OrgType.WHOLESALER),
        Organization(ext_id="ph", org_type=OrgType.LOCAL_ORG),
    ]
    locs = [
        Location(ext_id="loc_obp", org_ext_id="obp", market_code=market, postal_code="1000"),
        Location(ext_id="loc_wh", org_ext_id="wh", market_code=market, postal_code="2000"),
        Location(ext_id="loc_ph", org_ext_id="ph", market_code=market, postal_code="3000"),
    ]
    batches = [
        Batch(ext_id="b", product_ext_id="prod", manufacturer_org_ext_id="obp",
              intended_markets=(market,)),
    ]
    edges = [
        LocationEdge("loc_obp", "loc_wh", cost=1.0, capacity=10),
        LocationEdge("loc_wh", "loc_ph", cost=1.0, capacity=10),
    ]
    behavior = {
        "loc_ph": LocationBehavior(
            verify_prob=behavior_verify,
            decomission_prob=behavior_decom,
            reactivate_prob=behavior_react,
        )
    }
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=[_product()],
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location=behavior,
        seed=1,
    )


def _freeze_supply(inp) -> None:
    """Disable demand + supply policies so packs stay put and only behavior fires."""
    inp.location_demand_policy_id[:] = 0
    inp.location_supply_policy_id[:] = 0


def test_verify_in_foreign_market_raises_out_of_market_alert() -> None:
    pack = Pack(ext_id="p0", product_ext_id="prod", batch_ext_id="b", serial="SN0",
                initial_market_code="BG", initial_location_ext_id="loc_ph",
                initial_state=PackState.ACTIVE)
    inp = compile_scenario(_scenario([pack], behavior_verify=1.0))
    _freeze_supply(inp)
    # Move the pharmacy into a market the serial was never provisioned to.
    foreign = len(inp.market_code)
    inp.market_code.append("XX")
    inp.n_markets = len(inp.market_code)
    ph = inp.location_ext_id.index("loc_ph")
    inp.location_market_id[ph] = np.uint32(foreign)
    inp.pack_initial_market_id[0] = np.uint32(foreign)

    sim = create_native_simulator(inp)
    sim.run_ticks(5)
    records = events_as_records(sim, inp)
    alerts = [r for r in records if r["event_type"] == "ALERT"]
    assert any(r["alert_code"] == "OUT_OF_MARKET" for r in alerts)
    assert all(r["from_location"] == "loc_ph" for r in alerts if r["alert_code"] == "OUT_OF_MARKET")


def test_decommission_carries_reason_and_market() -> None:
    pack = Pack(ext_id="p0", product_ext_id="prod", batch_ext_id="b", serial="SN0",
                initial_market_code="BG", initial_location_ext_id="loc_ph",
                initial_state=PackState.ACTIVE)
    inp = compile_scenario(_scenario([pack], behavior_decom=1.0))
    _freeze_supply(inp)
    sim = create_native_simulator(inp)
    sim.run_ticks(3)
    records = events_as_records(sim, inp)
    decom = [r for r in records if r["event_type"] == "DECOMMISSION"]
    assert decom
    # Default site reason is DISPENSED; the event is scoped to the BG NMVS.
    assert decom[0]["decommission_reason"] == "DISPENSED"
    assert decom[0]["market"] == "BG"


def test_clone_makes_registry_diverge_and_raises_duplicate_alert() -> None:
    # Two physical packs, one serial: a clone. One sits at a no-behavior wholesaler (stays live),
    # the other is dispensed at the pharmacy -> the NMVS retires the shared serial while the clone
    # is still physically active, so registry and physical state diverge.
    packs = [
        Pack(ext_id="genuine", product_ext_id="prod", batch_ext_id="b", serial="SN0",
             initial_market_code="BG", initial_location_ext_id="loc_ph",
             initial_state=PackState.ACTIVE),
        Pack(ext_id="clone", product_ext_id="prod", batch_ext_id="b", serial="SN1",
             initial_market_code="BG", initial_location_ext_id="loc_wh",
             initial_state=PackState.ACTIVE),
    ]
    inp = compile_scenario(_scenario(packs, behavior_verify=1.0, behavior_decom=1.0))
    _freeze_supply(inp)
    # Forge a clone: point the wholesaler pack at the pharmacy pack's serial id (as a counterfeit
    # cloning a real serial would). The NMVS then holds a single record for two physical packs.
    inp.pack_serial_id[1] = inp.pack_serial_id[0]

    sim = create_native_simulator(inp)
    assert sim.registry_matches_physical()
    sim.run_ticks(5)
    # The pharmacy pack is decommissioned; the wholesaler clone remains physically live.
    assert not sim.registry_matches_physical()
    records = events_as_records(sim, inp)
    assert any(
        r["event_type"] == "ALERT" and r["alert_code"] == "DUPLICATE_SERIAL" for r in records
    )


def test_reactivation_window_is_enforced() -> None:
    pack = Pack(ext_id="p0", product_ext_id="prod", batch_ext_id="b", serial="SN0",
                initial_market_code="BG", initial_location_ext_id="loc_ph",
                initial_state=PackState.ACTIVE)

    def reactivate_events(window: int) -> int:
        inp = compile_scenario(_scenario([pack], behavior_decom=1.0, behavior_react=1.0))
        _freeze_supply(inp)
        inp.reactivation_window_ticks = window
        sim = create_native_simulator(inp)
        sim.run_ticks(20)
        records = events_as_records(sim, inp)
        return sum(1 for r in records if r["event_type"] == "REACTIVATE")

    # Window 0: every reactivation falls outside the reversal window and is rejected.
    assert reactivate_events(0) == 0
    # Generous window: reactivations are accepted and logged.
    assert reactivate_events(50) > 0
