from __future__ import annotations

from pathlib import Path

from compiler.compile import compile_scenario
from canonical.models import ActorRole, CanonicalDataset, CanonicalEdge, CanonicalNode
from canonical.normalize import canonicalize_bg_registry, canonicalize_spor, merge_nodes
from canonical.routing import build_routing_edges
from canonical.scenario_builder import build_scenario_from_canonical
from policy.scenarios import bulgaria_registry_scenario


def test_merge_nodes_enriches_coordinates() -> None:
    primary = [
        CanonicalNode(
            node_id="a",
            name="Node A",
            role=ActorRole.PHARMACY,
            market_code="BG",
            city="Sofia",
            address="Street 1",
            postal_code="1000",
            is_active=True,
        )
    ]
    secondary = [
        CanonicalNode(
            node_id="b",
            name="Node B",
            role=ActorRole.UNKNOWN,
            market_code="BG",
            city="Sofia",
            address="Street 1",
            postal_code="1000",
            is_active=True,
            latitude=42.7,
            longitude=23.3,
            source_ids={"spor.location_id": "LOC-1"},
        )
    ]
    merged = merge_nodes(primary, secondary)
    assert len(merged) == 1
    assert merged[0].latitude == 42.7
    assert "spor.location_id" in merged[0].source_ids


def test_build_routing_edges_for_supply_chain_roles() -> None:
    nodes = [
        CanonicalNode(
            node_id="m",
            name="Manufacturer",
            role=ActorRole.MANUFACTURER,
            market_code="BG",
            city="Sofia",
            address="A",
            postal_code="1000",
            is_active=True,
            latitude=42.6977,
            longitude=23.3219,
        ),
        CanonicalNode(
            node_id="w",
            name="Wholesaler",
            role=ActorRole.WHOLESALER,
            market_code="BG",
            city="Plovdiv",
            address="B",
            postal_code="4000",
            is_active=True,
            latitude=42.1354,
            longitude=24.7453,
        ),
        CanonicalNode(
            node_id="p",
            name="Pharmacy",
            role=ActorRole.PHARMACY,
            market_code="BG",
            city="Burgas",
            address="C",
            postal_code="8000",
            is_active=True,
            latitude=42.5048,
            longitude=27.4626,
        ),
    ]
    edges = build_routing_edges(nodes, k_neighbors=2, max_distance_km=500)
    assert edges
    assert any(e.src_node_id == "m" and e.dst_node_id == "w" for e in edges)
    assert any(e.src_node_id == "w" and e.dst_node_id == "p" for e in edges)


def test_build_scenario_from_canonical_compiles() -> None:
    dataset = CanonicalDataset(
        nodes=[
            CanonicalNode(
                node_id="m1",
                name="M1",
                role=ActorRole.MANUFACTURER,
                market_code="BG",
                city="Sofia",
                address="A",
                postal_code="1000",
                is_active=True,
                latitude=42.6977,
                longitude=23.3219,
            ),
            CanonicalNode(
                node_id="w1",
                name="W1",
                role=ActorRole.WHOLESALER,
                market_code="BG",
                city="Plovdiv",
                address="B",
                postal_code="4000",
                is_active=True,
                latitude=42.1354,
                longitude=24.7453,
            ),
            CanonicalNode(
                node_id="p1",
                name="P1",
                role=ActorRole.PHARMACY,
                market_code="BG",
                city="Burgas",
                address="C",
                postal_code="8000",
                is_active=True,
                latitude=42.5048,
                longitude=27.4626,
            ),
        ],
        edges=[
            CanonicalEdge(
                src_node_id="m1",
                dst_node_id="w1",
                distance_km=130.0,
                travel_minutes=140.0,
                route_cost=100.0,
                capacity_per_tick=500,
            ),
            CanonicalEdge(
                src_node_id="w1",
                dst_node_id="p1",
                distance_km=200.0,
                travel_minutes=220.0,
                route_cost=150.0,
                capacity_per_tick=200,
            ),
        ],
    )
    scenario = build_scenario_from_canonical(dataset, packs_per_obp=10, max_locations=10)
    inp = compile_scenario(scenario)
    assert inp.n_locations == 3
    assert inp.n_edges == 2
    assert inp.n_packs == 10


def test_bulgaria_registry_scenario_smoke() -> None:
    repo = Path(__file__).resolve().parent.parent
    scenario = bulgaria_registry_scenario(
        bg_registry_path=str(repo / "data" / "data.json"),
        spor_locations_path=str(repo / "data" / "spor_locations.csv"),
        geocode=False,
        max_locations=80,
        packs_per_obp=20,
    )
    inp = compile_scenario(scenario)
    assert inp.n_locations <= 80
    assert inp.n_packs > 0
    assert inp.n_edges >= 0


def test_canonicalizers_parse_minimal_records() -> None:
    bg_rows = [
        {
            "apteka_n": "1001",
            "c_name": "Аптека 1",
            "full_n": "АП-1",
            "a_town": "София",
            "a_address": "ул. Тест 1",
            "a_type": 2,
            "is_active": True,
        }
    ]
    spor_rows = [
        {
            "Location ID": "LOC-1",
            "Name": "Some Wholesaler",
            "Category Classification Category  Display Name": "Industry¦Pharmaceutical company",
            "Address Country Code": "BG",
            "Address City": "Sofia¦София",
            "Address Line 1": "ul. Test 1",
            "Address Postal Code": "1000",
            "Address GPS Location": "42.7, 23.3",
            "Status": "ACTIVE",
        }
    ]
    bg_nodes = canonicalize_bg_registry(bg_rows)
    spor_nodes = canonicalize_spor(spor_rows)
    assert bg_nodes and spor_nodes
    assert bg_nodes[0].role == ActorRole.PHARMACY
    assert spor_nodes[0].role == ActorRole.WHOLESALER
