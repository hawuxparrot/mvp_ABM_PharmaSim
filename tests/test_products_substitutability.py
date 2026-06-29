from __future__ import annotations

from pathlib import Path

import pytest

from canonical.loaders import (
    canonicalize_products,
    load_product_rows,
    product_intended_markets,
    product_substitution_groups,
)
from canonical.substitutability import SubstitutabilityIndex, atc_prefix
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


def _scenario_with_products(products: list[Product]) -> Scenario:
    """Minimal compilable scenario seeding one pack per catalog product at a wholesaler."""
    orgs = [
        Organization(ext_id="obp", org_type=OrgType.OBP),
        Organization(ext_id="wh", org_type=OrgType.WHOLESALER),
        Organization(ext_id="ph", org_type=OrgType.LOCAL_ORG),
    ]
    locs = [
        Location(ext_id="loc_obp", org_ext_id="obp", market_code="BG", postal_code="1000"),
        Location(ext_id="loc_wh", org_ext_id="wh", market_code="BG", postal_code="2000"),
        Location(ext_id="loc_ph", org_ext_id="ph", market_code="BG", postal_code="3000"),
    ]
    batches = [
        Batch(
            ext_id=f"batch_{p.ext_id}",
            product_ext_id=p.ext_id,
            manufacturer_org_ext_id="obp",
            intended_markets=("BG",),
        )
        for p in products
    ]
    packs = [
        Pack(
            ext_id=f"pack_{i}",
            product_ext_id=p.ext_id,
            batch_ext_id=f"batch_{p.ext_id}",
            serial=f"SN{i:06d}",
            initial_market_code="BG",
            initial_location_ext_id="loc_wh",
            initial_state=PackState.ACTIVE,
        )
        for i, p in enumerate(products)
    ]
    edges = [
        LocationEdge("loc_obp", "loc_wh", cost=1.0, capacity=10),
        LocationEdge("loc_wh", "loc_ph", cost=1.0, capacity=10),
    ]
    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location={},
        seed=3,
    )


def _p(ext_id: str, atc: str) -> Product:
    return Product(
        ext_id=ext_id,
        codes=(ProductCode(scheme=ProductCodeScheme.GTIN, value=ext_id, is_primary=True),),
        atc_code=atc,
    )


def test_atc_prefix_levels() -> None:
    assert atc_prefix("C10AA05", 1) == "C"
    assert atc_prefix("C10AA05", 2) == "C10"
    assert atc_prefix("C10AA05", 3) == "C10A"
    assert atc_prefix("C10AA05", 4) == "C10AA"
    assert atc_prefix("C10AA05", 5) == "C10AA05"
    with pytest.raises(ValueError):
        atc_prefix("C10AA05", 6)


def test_substitutes_same_substance_at_level5() -> None:
    products = [
        _p("ator20", "C10AA05"),
        _p("ator40", "C10AA05"),
        _p("simva20", "C10AA01"),
        _p("amox", "J01CA04"),
    ]
    idx = SubstitutabilityIndex(products, default_level=5)
    # ATC5: only the other atorvastatin strength shares the active substance.
    assert idx.substitutes("ator20") == ["ator40"]
    assert not idx.are_substitutes("ator20", "simva20", level=5)


def test_substitutes_same_subgroup_at_level4() -> None:
    products = [
        _p("ator20", "C10AA05"),
        _p("ator40", "C10AA05"),
        _p("simva20", "C10AA01"),
        _p("rosuva", "C10AA07"),
        _p("amox", "J01CA04"),
    ]
    idx = SubstitutabilityIndex(products, default_level=4)
    subs = set(idx.substitutes("ator20", level=4))
    # ATC4 C10AA = all statins, excluding the antibiotic and self.
    assert subs == {"ator40", "simva20", "rosuva"}
    assert "amox" not in subs


def test_level_cutoff_widens_with_lower_level() -> None:
    products = [
        _p("ator20", "C10AA05"),
        _p("simva20", "C10AA01"),
        _p("ezetimibe", "C10AX09"),  # same ATC3 (C10A) but different ATC4
    ]
    idx = SubstitutabilityIndex(products)
    assert idx.substitutes("ator20", level=4) == ["simva20"]
    assert set(idx.substitutes("ator20", level=3)) == {"simva20", "ezetimibe"}


def test_manual_group_override_unions_with_atc() -> None:
    products = [
        _p("ator20", "C10AA05"),
        _p("special", "X99XX99"),  # unrelated ATC
    ]
    idx = SubstitutabilityIndex(
        products,
        default_level=5,
        group_overrides={"ator20": "grpA", "special": "grpA"},
    )
    assert idx.are_substitutes("ator20", "special")
    assert "special" in idx.substitutes("ator20")


def test_catalog_csv_loads_and_indexes() -> None:
    repo = Path(__file__).resolve().parent.parent
    rows = load_product_rows(repo / "data" / "products.csv")
    products = canonicalize_products(rows)
    assert len(products) == len(rows) >= 10
    assert all(p.atc_code for p in products)

    markets = product_intended_markets(rows)
    assert markets["prod_atorvastatin_20"] == ("BG",)
    # no manual overrides declared in the shipped catalog
    assert product_substitution_groups(rows) == {}

    idx = SubstitutabilityIndex(products, default_level=5)
    assert "prod_atorvastatin_40" in idx.substitutes("prod_atorvastatin_20")


def test_apply_product_demand_weights_by_atc() -> None:
    from compiler.compile import compile_scenario
    from runtime.experiments import apply_product_demand_weights

    repo = Path(__file__).resolve().parent.parent
    rows = load_product_rows(repo / "data" / "products.csv")
    products = canonicalize_products(rows)
    scenario = _scenario_with_products(products)
    inp = compile_scenario(scenario)

    # Statins (C10) get a high base rate; antibiotics (J01) low; everything else default.
    apply_product_demand_weights(
        inp,
        scenario,
        weight_by_atc_prefix={"C10": 5.0, "J01": 0.5},
        default_weight=1.0,
    )
    by_id = {p.ext_id: i for i, p in enumerate(scenario.products)}
    w = inp.product_demand_weight
    assert w[by_id["prod_atorvastatin_20"]] == 5.0
    assert w[by_id["prod_amoxicillin_500"]] == 0.5
    assert w[by_id["prod_paracetamol_500"]] == 1.0
