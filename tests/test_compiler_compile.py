"""Compile pipeline: Scenario -> EngineInput."""

import numpy as np
import pytest

from compiler.compile import compile_scenario
from compiler.enums import decode_org_type_u8, decode_pack_state_u8
from compiler.types import ENGINE_INPUT_SCHEMA_VERSION
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
    assert inp.location_has_behavior is not None
    assert inp.location_verify_prob is not None
    n = inp.n_locations
    assert inp.location_has_behavior.shape == (n,)
    assert float(inp.location_verify_prob[inp.location_ext_id.index("loc_wh_de")]) == pytest.approx(0.2)
