"""Compile a validated policy.model,s.Scenario into compiler.types.EngineInput."""

from __future__ import annotations

import numpy as np

from compiler.enums import org_type_u8, pack_state_u8
from compiler.types import ENGINE_INPUT_SCHEMA_VERSION, EngineInput
from compiler.validate import validate_scenario
from policy.models import Scenario


def _collect_market_codes(s: Scenario) -> list[str]:
    codes: set[str] = set()
    for loc in s.locations:
        codes.add(loc.market_code)
    for pk in s.packs:
        codes.add(pk.initial_market_code)
    for b in s.batches:
        for m in b.intended_markets:
            codes.add(m)
    return sorted(codes)


def compile_scenario(scenario: Scenario) -> EngineInput:
    """
    Validate scenario, assign dense IDs (list order = row id), build columnar EngineInput.

    ID policy: for each entity list on Scenario, row_index == ext_id space dense id (0 .. n-1). 
    dense id (0 .. n-1). Markets are interned: sorted unique market strings → market_id.
    """
    s = validate_scenario(scenario)

    market_code = _collect_market_codes(s)
    market_str_to_id = {m: i for i, m in enumerate(market_code)}
    n_markets = len(market_code)

    org_ext_to_id = {o.ext_id: i for i, o in enumerate(s.organizations)}
    loc_ext_to_id = {loc.ext_id: i for i, loc in enumerate(s.locations)}
    product_ext_to_id = {p.ext_id: i for i, p in enumerate(s.products)}
    batch_ext_to_id = {b.ext_id: i for i, b in enumerate(s.batches)}

    n_org = len(s.organizations)
    n_loc = len(s.locations)
    n_prod = len(s.products)
    n_batch = len(s.batches)
    n_pack = len(s.packs)

    org_type = np.zeros(n_org, dtype=np.uint8)
    org_ext_id = [o.ext_id for o in s.organizations]
    for i, o in enumerate(s.organizations):
        org_type[i] = org_type_u8(o.org_type)

    location_org_id = np.zeros(n_loc, dtype=np.uint32)
    location_market_id = np.zeros(n_loc, dtype=np.uint32)
    location_ext_id = [loc.ext_id for loc in s.locations]
    for i, loc in enumerate(s.locations):
        location_org_id[i] = org_ext_to_id[loc.org_ext_id]
        location_market_id[i] = market_str_to_id[loc.market_code]

    batch_product_id = np.zeros(n_batch, dtype=np.uint32)
    batch_manufacturer_org_id = np.zeros(n_batch, dtype=np.uint32)
    batch_ext_id = [b.ext_id for b in s.batches]
    flat_markets: list[int] = []
    offset: list[int] = [0]
    for b in s.batches:
        for m in b.intended_markets:
            flat_markets.append(market_str_to_id[m])
        offset.append(len(flat_markets))

    batch_intended_market_offset = np.array(offset, dtype=np.uint32)
    batch_intended_market_id = np.array(flat_markets, dtype=np.uint32)

    for i, b in enumerate(s.batches):
        batch_product_id[i] = product_ext_to_id[b.product_ext_id]
        batch_manufacturer_org_id[i] = org_ext_to_id[b.manufacturer_org_ext_id]

    pack_product_id = np.zeros(n_pack, dtype=np.uint32)
    pack_batch_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_location_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_market_id = np.zeros(n_pack, dtype=np.uint32)
    pack_initial_state = np.zeros(n_pack, dtype=np.uint8)
    pack_serial: list[str] = []
    pack_ext_id = [p.ext_id for p in s.packs]

    for i, p in enumerate(s.packs):
        pack_product_id[i] = product_ext_to_id[p.product_ext_id]
        pack_batch_id[i] = batch_ext_to_id[p.batch_ext_id]
        pack_initial_location_id[i] = loc_ext_to_id[p.initial_location_ext_id]
        pack_initial_market_id[i] = market_str_to_id[p.initial_market_code]
        pack_initial_state[i] = pack_state_u8(p.initial_state)
        pack_serial.append(p.serial)

    location_has_behavior: np.ndarray | None = None
    location_verify_prob: np.ndarray | None = None
    location_decommission_prob: np.ndarray | None = None
    location_reactivate_prob: np.ndarray | None = None

    if s.behavior_by_location:
        location_has_behavior = np.zeros(n_loc, dtype=np.uint8)
        location_verify_prob = np.zeros(n_loc, dtype=np.float32)
        location_decommission_prob = np.zeros(n_loc, dtype=np.float32)
        location_reactivate_prob = np.zeros(n_loc, dtype=np.float32)
        for i, loc in enumerate(s.locations):
            beh = s.behavior_by_location.get(loc.ext_id)
            if beh is None:
                continue
            location_has_behavior[i] = 1
            location_verify_prob[i] = np.float32(beh.verify_prob)
            location_decommission_prob[i] = np.float32(beh.decomission_prob)
            location_reactivate_prob[i] = np.float32(beh.reactivate_prob)

    engine_input = EngineInput(
        schema_version=ENGINE_INPUT_SCHEMA_VERSION,
        seed=s.seed,
        n_organizations=n_org,
        n_locations=n_loc,
        n_products=n_prod,
        n_batches=n_batch,
        n_packs=n_pack,
        n_markets=n_markets,
        market_code=market_code,
        org_type=org_type,
        org_ext_id=org_ext_id,
        location_org_id=location_org_id,
        location_market_id=location_market_id,
        location_ext_id=location_ext_id,
        batch_product_id=batch_product_id,
        batch_manufacturer_org_id=batch_manufacturer_org_id,
        batch_intended_market_offset=batch_intended_market_offset,
        batch_intended_market_id=batch_intended_market_id,
        batch_ext_id=batch_ext_id,
        pack_product_id=pack_product_id,
        pack_batch_id=pack_batch_id,
        pack_initial_location_id=pack_initial_location_id,
        pack_initial_market_id=pack_initial_market_id,
        pack_initial_state=pack_initial_state,
        pack_serial=pack_serial,
        pack_ext_id=pack_ext_id,
        location_has_behavior=location_has_behavior,
        location_verify_prob=location_verify_prob,
        location_decommission_prob=location_decommission_prob,
        location_reactivate_prob=location_reactivate_prob,
    )
    engine_input.validate_shapes()
    return engine_input
