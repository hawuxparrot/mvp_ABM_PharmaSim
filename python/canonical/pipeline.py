from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from analytics.fraud import (
    inject_cross_market_anomalies,
    inject_volume_spike_anomalies,
)
from policy.models import Scenario
from policy.transactions import AnomalyLabel, TransactionPlan

from .geocoding import (
    CachedGeocoder,
    JsonGeocodeCache,
    NominatimGeocoder,
    geocode_nodes,
)
from .loaders import load_bg_registry_rows, load_spor_rows
from .models import CanonicalDataset
from .normalize import canonicalize_bg_registry, canonicalize_spor, merge_nodes
from .routing import build_routing_edges
from .scenario_builder import build_scenario_from_canonical
from .transactions import build_synthetic_transaction_plan


@dataclass(frozen=True)
class BulgariaExperimentBundle:
    scenario: Scenario
    canonical: CanonicalDataset
    transaction_plan: TransactionPlan
    anomaly_labels: list[AnomalyLabel]


def build_canonical_dataset(
    *,
    bg_registry_path: str | Path,
    spor_locations_path: str | Path,
    geocode: bool = False,
    geocode_cache_path: str | Path | None = None,
    geocode_max_new_requests: int | None = None,
) -> CanonicalDataset:
    registry_rows = load_bg_registry_rows(bg_registry_path)
    spor_bg_rows = load_spor_rows(spor_locations_path, country_code="BG")

    bg_nodes = canonicalize_bg_registry(registry_rows)
    spor_nodes = canonicalize_spor(spor_bg_rows)
    merged = merge_nodes(bg_nodes, spor_nodes)

    unresolved: list[str] = []
    if geocode:
        cache_path = (
            Path(geocode_cache_path)
            if geocode_cache_path is not None
            else Path("data/geocode_cache_bg.json")
        )
        geocoder = CachedGeocoder(
            provider=NominatimGeocoder(),
            cache=JsonGeocodeCache(cache_path),
        )
        merged, unresolved = geocode_nodes(
            merged,
            geocoder,
            max_new_requests=geocode_max_new_requests,
        )

    edges = build_routing_edges(merged)
    notes = [
        f"bg_registry_active_nodes={len(bg_nodes)}",
        f"spor_bg_nodes={len(spor_nodes)}",
        f"merged_nodes={len(merged)}",
        f"routing_edges={len(edges)}",
    ]
    return CanonicalDataset(
        nodes=merged,
        edges=edges,
        unresolved_geocodes=unresolved,
        notes=notes,
    )


def build_bulgaria_scenario(
    *,
    bg_registry_path: str | Path = "data/data.json",
    spor_locations_path: str | Path = "data/spor_locations.csv",
    geocode: bool = False,
    geocode_cache_path: str | Path | None = None,
    geocode_max_new_requests: int | None = None,
    seed: int = 42,
    packs_per_obp: int = 250,
    max_locations: int = 350,
) -> tuple[Scenario, CanonicalDataset]:
    canonical = build_canonical_dataset(
        bg_registry_path=bg_registry_path,
        spor_locations_path=spor_locations_path,
        geocode=geocode,
        geocode_cache_path=geocode_cache_path,
        geocode_max_new_requests=geocode_max_new_requests,
    )
    scenario = build_scenario_from_canonical(
        canonical,
        seed=seed,
        packs_per_obp=packs_per_obp,
        max_locations=max_locations,
    )
    return scenario, canonical


def build_bulgaria_experiment_bundle(
    *,
    bg_registry_path: str | Path = "data/data.json",
    spor_locations_path: str | Path = "data/spor_locations.csv",
    geocode: bool = False,
    geocode_cache_path: str | Path | None = None,
    geocode_max_new_requests: int | None = None,
    seed: int = 42,
    packs_per_obp: int = 250,
    max_locations: int = 350,
    tx_horizon_ticks: int = 30,
    tx_order_lambda_per_edge: float = 0.6,
    tx_max_units_per_order: int = 40,
    inject_volume_spikes: bool = True,
    inject_cross_market: bool = True,
) -> BulgariaExperimentBundle:
    """
    Build a Phase B-ready experiment bundle:
    scenario + canonical dataset + synthetic transaction plan + anomaly labels.
    """
    scenario, canonical = build_bulgaria_scenario(
        bg_registry_path=bg_registry_path,
        spor_locations_path=spor_locations_path,
        geocode=geocode,
        geocode_cache_path=geocode_cache_path,
        geocode_max_new_requests=geocode_max_new_requests,
        seed=seed,
        packs_per_obp=packs_per_obp,
        max_locations=max_locations,
    )
    plan = build_synthetic_transaction_plan(
        scenario,
        horizon_ticks=tx_horizon_ticks,
        order_lambda_per_edge=tx_order_lambda_per_edge,
        max_units_per_order=tx_max_units_per_order,
        seed=seed,
    )
    labels: list[AnomalyLabel] = []
    if inject_volume_spikes:
        plan, volume_labels = inject_volume_spike_anomalies(plan, seed=seed + 1)
        labels.extend(volume_labels)
    if inject_cross_market:
        plan, market_labels = inject_cross_market_anomalies(plan, seed=seed + 2)
        labels.extend(market_labels)
    return BulgariaExperimentBundle(
        scenario=scenario,
        canonical=canonical,
        transaction_plan=plan,
        anomaly_labels=labels,
    )

