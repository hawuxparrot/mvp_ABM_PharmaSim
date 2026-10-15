from __future__ import annotations

import pytest

from canonical.models import ActorRole, CanonicalDataset, CanonicalNode
from compiler.compile import compile_scenario
from learning.case_builder import CaseBuilder, CaseBuilderConfig
from policy.models import (
    Batch,
    Location,
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


def _fixture_builder(config: CaseBuilderConfig | None = None) -> tuple[CaseBuilder, dict[str, int]]:
    orgs = [
        Organization("org_a", OrgType.LOCAL_ORG),
        Organization("org_b", OrgType.LOCAL_ORG),
        Organization("org_h1", OrgType.LOCAL_ORG),
        Organization("org_h2", OrgType.LOCAL_ORG),
    ]
    locs = [
        Location("loc_a", "org_a", "BG", "1000"),
        Location("loc_b", "org_b", "BG", "1000"),
        Location("loc_h1", "org_h1", "BG", "2000"),
        Location("loc_h2", "org_h2", "BG", "3000"),
    ]
    product = Product("prod", (ProductCode(ProductCodeScheme.GTIN, "00000000000001", True),))
    batch = Batch("batch", "prod", "org_a", ("BG",))
    packs = [
        Pack("pack_0", "prod", "batch", "SN0000", "BG", "loc_a", PackState.ACTIVE),
    ]
    scenario = Scenario(
        organizations=orgs,
        locations=locs,
        products=[product],
        batches=[batch],
        packs=packs,
        location_edges=[
            LocationEdge("loc_a", "loc_b", 1.0, 10),
            LocationEdge("loc_h1", "loc_h2", 1.0, 10),
        ],
        seed=7,
    )
    canonical = CanonicalDataset(
        nodes=[
            CanonicalNode("a", "A", ActorRole.PHARMACY, "BG", "Sofia", "", "1000", True, 42.0, 23.0),
            CanonicalNode("b", "B", ActorRole.PHARMACY, "BG", "Sofia", "", "1000", True, 42.1, 23.0),
            CanonicalNode("h1", "H1", ActorRole.HOSPITAL, "BG", "Plovdiv", "", "2000", True, 42.2, 24.7),
            CanonicalNode("h2", "H2", ActorRole.HOSPITAL, "BG", "Varna", "", "3000", True, 43.2, 27.9),
        ],
        edges=[],
    )
    inp = compile_scenario(scenario)
    ids = {name: inp.location_ext_id.index(f"loc_{name}") for name in ("a", "b", "h1", "h2")}
    return CaseBuilder(inp, scenario, canonical, config=config), ids


def _move_event(src: int, dst: int) -> dict[str, object]:
    return {
        "tick": 5,
        "pack_id": 0,
        "event_type": "MOVE",
        "from_location_id": src,
        "to_location_id": dst,
        "alert_code": None,
        "decommission_reason": None,
    }


def test_same_location_feature_extraction() -> None:
    builder, ids = _fixture_builder()
    cases = builder.build([_move_event(ids["a"], ids["a"])], builder.new_history())
    assert cases[0].same_location is True
    assert cases[0].same_organization is True
    assert cases[0].distance_km == pytest.approx(0.0)


def test_same_postal_code_feature_extraction() -> None:
    builder, ids = _fixture_builder()
    cases = builder.build([_move_event(ids["a"], ids["b"])], builder.new_history())
    assert cases[0].same_postal_code is True
    assert cases[0].same_city is True
    assert cases[0].from_role == "PHARMACY"
    assert cases[0].to_role == "PHARMACY"


def test_distance_feature_extraction() -> None:
    builder, ids = _fixture_builder()
    cases = builder.build([_move_event(ids["a"], ids["b"])], builder.new_history())
    assert cases[0].distance_km is not None
    assert cases[0].distance_km > 10.0


def test_deterministic_safe_event_sampling() -> None:
    config = CaseBuilderConfig(safe_sample_rate=1.0, safe_sample_seed=123)
    builder, ids = _fixture_builder(config)
    event = {
        "tick": 9,
        "pack_id": 0,
        "event_type": "VERIFY",
        "from_location_id": None,
        "to_location_id": ids["a"],
        "alert_code": None,
        "decommission_reason": None,
    }
    first = builder.build([event], builder.new_history(), stream_id="demo")
    second = builder.build([event], builder.new_history(), stream_id="demo")
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert first and first[0].event_type == "VERIFY"
