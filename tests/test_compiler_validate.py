"""Tests for compiler scenario validation. AI generated tests."""

import pytest

from compiler.validate import ScenarioValidationError, validate_scenario
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


def test_two_markets_demo_passes() -> None:
    from policy.scenarios import two_markets_demo

    s = two_markets_demo()
    out = validate_scenario(s)
    assert out is s


def test_duplicate_pack_ext_id_fails() -> None:
    orgs = [Organization(ext_id="o1", org_type=OrgType.OBP)]
    locs = [Location(ext_id="l1", org_ext_id="o1", market_code="DE", postal_code="x")]
    p = Product(
        ext_id="p1",
        codes=(ProductCode(ProductCodeScheme.GTIN, "1", True),),
    )
    b = Batch(
        ext_id="b1",
        product_ext_id="p1",
        manufacturer_org_ext_id="o1",
        intended_markets=("DE",),
    )
    packs = [
        Pack("pk1", "p1", "b1", "S1", "DE", "l1"),
        Pack("pk1", "p1", "b1", "S2", "DE", "l1"),
    ]
    s = Scenario(organizations=orgs, locations=locs, products=[p], batches=[b], packs=packs)
    with pytest.raises(ScenarioValidationError, match="duplicate ext_id"):
        validate_scenario(s)


def test_unknown_product_on_pack_fails() -> None:
    orgs = [Organization(ext_id="o1", org_type=OrgType.OBP)]
    locs = [Location(ext_id="l1", org_ext_id="o1", market_code="DE", postal_code="x")]
    p = Product(
        ext_id="p1",
        codes=(ProductCode(ProductCodeScheme.GTIN, "1", True),),
    )
    b = Batch(
        ext_id="b1",
        product_ext_id="p1",
        manufacturer_org_ext_id="o1",
        intended_markets=("DE",),
    )
    packs = [
        Pack(
            ext_id="pk1",
            product_ext_id="missing_product",
            batch_ext_id="b1",
            serial="S1",
            initial_market_code="DE",
            initial_location_ext_id="l1",
            initial_state=PackState.UPLOADED,
        ),
    ]
    s = Scenario(organizations=orgs, locations=locs, products=[p], batches=[b], packs=packs)
    with pytest.raises(ScenarioValidationError, match="unknown product_ext_id"):
        validate_scenario(s)
