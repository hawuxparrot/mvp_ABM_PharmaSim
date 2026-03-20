from __future__ import annotations
from policy.models import (
    Batch,
    Location,
    LocationBehavior,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario
)

def two_markets_demo() -> Scenario:
    """
    Small demo scenario: 1 OBP (DE), 1 wholesaler + 1 pharmacy in DE, 1 wholesaler + 1 pharmacy in FR
    1 product (GTIN), 1 batch, a few packs starting in DE
    """

    orgs = [
        Organization(ext_id="obp_de", org_type=OrgType.OBP),
        Organization(ext_id='wh_de', org_type=OrgType.WHOLESALER),
        Organization(ext_id='ph_de', org_type=OrgType.LOCAL_ORG),
        Organization(ext_id='wh_fr', org_type=OrgType.WHOLESALER),
        Organization(ext_id='ph_fr', org_type=OrgType.LOCAL_ORG),
        Organization(ext_id='nmvo_de', org_type=OrgType.NMVO),
        Organization(ext_id='nmvo_fr', org_type=OrgType.NMVO),
        Organization(ext_id='emvo', org_type=OrgType.EMVO),
    ]

    locations = [
        Location(ext_id='loc_obp_de', org_ext_id='obp_de', market_code='DE', postal_code='11451'),
        Location(ext_id='loc_wh_de', org_ext_id='wh_de', market_code='DE', postal_code='51334'),
        Location(ext_id='loc_ph_de', org_ext_id='ph_de', market_code='DE', postal_code='45431'),
        Location(ext_id='loc_wh_fr', org_ext_id='wh_fr', market_code='FR', postal_code='85439'),
        Location(ext_id='loc_ph_fr', org_ext_id='ph_fr', market_code='FR', postal_code='98046'),
        Location(ext_id='loc_nmvo_de', org_ext_id='nmvo_de', market_code='DE', postal_code='90541'),
        Location(ext_id='loc_nmvo_fr', org_ext_id='nmvo_fr', market_code='FR', postal_code='32445'),
        Location(ext_id='loc_emvo', org_ext_id='emvo', market_code='EU', postal_code='52439'),
    ]

    products = [
        Product(ext_id='prod_1', codes=[ProductCode(scheme=ProductCodeScheme.GTIN, value='01234567891011', is_primary=True)])
    ]

    batches = [
        Batch(
            ext_id="batch_001",
            product_ext_id="prod_amox_500",
            manufacturer_org_ext_id="obp_acme",
            intended_markets=("DE", "FR"),
        ),
    ]
    packs = [
        Pack(
            ext_id="pack_001",
            product_ext_id="prod_amox_500",
            batch_ext_id="batch_001",
            serial="SN000001",
            initial_market_code="DE",
            initial_location_ext_id="loc_obp_de",
            initial_state=PackState.UPLOADED,
        ),
        Pack(
            ext_id="pack_002",
            product_ext_id="prod_amox_500",
            batch_ext_id="batch_001",
            serial="SN000002",
            initial_market_code="DE",
            initial_location_ext_id="loc_obp_de",
            initial_state=PackState.ACTIVE,
        ),
    ]
    behavior = {
        "loc_wh_de": LocationBehavior(
            verify_prob=0.2,
            decomission_prob=0.05,
            reactivate_prob=0.01,
        ),
        "loc_ph_de": LocationBehavior(
            verify_prob=0.5,
            decomission_prob=0.15,
            reactivate_prob=0.02,
        ),
    }
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        behavior_by_location=behavior,
        seed=42,
    )