"""Compile pipeline: Scenario -> EngineInput."""

import numpy as np
import pytest

from compiler.compile import compile_scenario
from compiler.enums import decode_org_type_u8, decode_pack_state_u8
from compiler.types import ENGINE_INPUT_SCHEMA_VERSION
from policy.models import (
    Batch,
    Location,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario,
)
from policy.scenarios import two_markets_demo


def test_compile_two_markets_demo_passes_validate_shapes() -> None:
    inp = compile_scenario(two_markets_demo())
    inp.validate_shapes()
    assert inp.schema_version == ENGINE_INPUT_SCHEMA_VERSION


def test_compile_is_deterministic() -> None:
    s = two_markets_demo()
    a = compile_scenario(s)
    b = compile_scenario(s)
    assert a.seed == b.seed
    np.testing.assert_array_equal(a.org_type, b.org_type)
    np.testing.assert_array_equal(a.pack_initial_state, b.pack_initial_state)
    np.testing.assert_array_equal(a.batch_intended_market_offset, b.batch_intended_market_offset)
    assert a.pack_serial == b.pack_serial
    assert a.market_code == b.market_code


def test_compile_csr_intended_markets() -> None:
    inp = compile_scenario(two_markets_demo())
    off = inp.batch_intended_market_offset
    flat = inp.batch_intended_market_id
    assert off[0] == 0
    assert int(off[-1]) == len(flat)
    assert np.all(off[:-1] <= off[1:])
    # one batch in demo: DE, FR -> two market ids
    assert len(flat) == 2
    de_id = inp.market_code.index("DE")
    fr_id = inp.market_code.index("FR")
    assert int(flat[0]) == de_id and int(flat[1]) == fr_id


def test_compile_org_and_pack_columns_match_enums() -> None:
    inp = compile_scenario(two_markets_demo())
    for i in range(inp.n_organizations):
        decode_org_type_u8(int(inp.org_type[i]))
    for i in range(inp.n_packs):
        decode_pack_state_u8(int(inp.pack_initial_state[i]))


def test_compile_location_behavior_dense() -> None:
    inp = compile_scenario(two_markets_demo())
    n = inp.n_locations
    assert inp.location_has_behavior.shape == (n,)
    assert inp.location_verify_prob.shape == (n,)
    assert float(inp.location_verify_prob[inp.location_ext_id.index("loc_wh_de")]) == pytest.approx(0.2)


def test_compile_location_behavior_all_zeros_without_policy() -> None:
    """Option B: behavior columns are always present; all zeros when behavior_by_location is empty."""
    orgs = [Organization(ext_id="o1", org_type=OrgType.OBP)]
    locs = [Location(ext_id="l1", org_ext_id="o1", market_code="DE", postal_code="x")]
    prod = Product(
        ext_id="p1",
        codes=(ProductCode(ProductCodeScheme.GTIN, "1", True),),
    )
    batch = Batch(
        ext_id="b1",
        product_ext_id="p1",
        manufacturer_org_ext_id="o1",
        intended_markets=("DE",),
    )
    packs = [
        Pack(
            ext_id="pk1",
            product_ext_id="p1",
            batch_ext_id="b1",
            serial="S1",
            initial_market_code="DE",
            initial_location_ext_id="l1",
            initial_state=PackState.UPLOADED,
        ),
    ]
    s = Scenario(
        organizations=orgs,
        locations=locs,
        products=[prod],
        batches=[batch],
        packs=packs,
        behavior_by_location={},
        seed=1,
    )
    inp = compile_scenario(s)
    assert inp.n_locations == 1
    np.testing.assert_array_equal(inp.location_has_behavior, np.array([0], dtype=np.uint8))
    np.testing.assert_array_equal(inp.location_verify_prob, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(inp.location_decommission_prob, np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(inp.location_reactivate_prob, np.array([0.0], dtype=np.float32))
