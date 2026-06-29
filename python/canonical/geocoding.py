from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from .models import CanonicalNode


@dataclass(frozen=True)
class GeocodeResult:
    latitude: float
    longitude: float
    confidence: float


class Geocoder(Protocol):
    def geocode(self, query: str) -> GeocodeResult | None: ...


class JsonGeocodeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cache: dict[str, GeocodeResult] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for query, payload in raw.items():
                self._cache[query] = GeocodeResult(
                    latitude=float(payload["latitude"]),
                    longitude=float(payload["longitude"]),
                    confidence=float(payload["confidence"]),
                )

    def get(self, query: str) -> GeocodeResult | None:
        return self._cache.get(query)

    def set(self, query: str, result: GeocodeResult) -> None:
        self._cache[query] = result
        serializable = {
            q: {
                "latitude": v.latitude,
                "longitude": v.longitude,
                "confidence": v.confidence,
            }
            for q, v in self._cache.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class CachedGeocoder:
    def __init__(self, provider: Geocoder, cache: JsonGeocodeCache):
        self.provider = provider
        self.cache = cache

    def geocode(self, query: str) -> GeocodeResult | None:
        hit = self.cache.get(query)
        if hit is not None:
            return hit
        fresh = self.provider.geocode(query)
        if fresh is not None:
            self.cache.set(query, fresh)
        return fresh


class NominatimGeocoder:
    def __init__(
        self,
        *,
        user_agent: str = "pharmasim-canonical/0.1",
        endpoint: str = "https://nominatim.openstreetmap.org/search",
        pause_seconds: float = 1.0,
        timeout_seconds: float = 20.0,
    ):
        self.user_agent = user_agent
        self.endpoint = endpoint
        self.pause_seconds = pause_seconds
        self.timeout_seconds = timeout_seconds

    def geocode(self, query: str) -> GeocodeResult | None:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "jsonv2",
                "limit": "1",
            }
        )
        req = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"User-Agent": self.user_agent},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        time.sleep(self.pause_seconds)
        if not payload:
            return None
        row = payload[0]
        return GeocodeResult(
            latitude=float(row["lat"]),
            longitude=float(row["lon"]),
            confidence=1.0,
        )


def build_geocode_query(node: CanonicalNode) -> str:
    parts = [node.address, node.city, node.postal_code, "Bulgaria"]
    return ", ".join(p for p in parts if p)


def build_geocode_fallback_query(node: CanonicalNode) -> str:
    """Coarser query (city centroid) for nodes whose full address fails to resolve.

    Most BG-registry pharmacies carry no postal code and noisy street addresses, so a
    city-level fallback recovers usable coordinates and, because the cached query is
    shared across all nodes in the same city, costs at most one request per city.
    """
    parts = [node.city, "Bulgaria"]
    return ", ".join(p for p in parts if p)


# BG registry uses verbose resort labels and occasional Latin transliterations that
# Nominatim does not match; map them to OSM-friendly queries (shared across many nodes).
_RESORT_CITY_ALIASES: dict[str, tuple[str, ...]] = {
    'курортен комплекс "слънчев бряг"': (
        "Sunny Beach, Nesebar, Bulgaria",
        "Слънчев бряг, Nesebar, Bulgaria",
        "Слънчев бряг, Bulgaria",
    ),
    'курортен комплекс "златни пясъци"': (
        "Golden Sands, Varna, Bulgaria",
        "Златни пясъци, Bulgaria",
    ),
    'курортен комплекс "пампорово"': ("Pamporovo, Bulgaria",),
    'курортен комплекс "св. св. константин и елена"': (
        "Saints Constantine and Helena, Varna, Bulgaria",
    ),
    'курортен комплекс "албена"': ("Albena, Dobrich, Bulgaria",),
    'курортен комплекс "боровец"': ("Borovets, Samokov, Bulgaria",),
}

_CITY_ALIASES: dict[str, str] = {
    "kirdzhali": "Kardzhali, Bulgaria",
    "parvomaj": "Parvomay, Bulgaria",
    "bouzhurishte": "Bozhurishte, Bulgaria",
    "stanboliyski": "Stamboliyski, Bulgaria",
    "turnicheni": "Tarnicheni, Bulgaria",
}


def _norm_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def build_geocode_queries(
    node: CanonicalNode,
    *,
    use_city_fallback: bool = True,
) -> list[str]:
    """Ordered deduplicated queries from specific to coarse."""
    import re

    seen: set[str] = set()
    out: list[str] = []

    def add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    add(build_geocode_query(node))
    if node.address:
        add(", ".join(p for p in [node.address, "Bulgaria"] if p))
        for match in re.finditer(r"к\.?\s*к\.?\s*([^,]+)", node.address, flags=re.IGNORECASE):
            resort = match.group(1).strip()
            add(f"{resort}, Bulgaria")
            add(f"{resort}, Nesebar, Bulgaria")
        for match in re.finditer(
            r"общ(?:ина|\.)\s*([^,]+)", node.address, flags=re.IGNORECASE
        ):
            municipality = match.group(1).strip()
            add(f"{municipality}, Bulgaria")

    if use_city_fallback:
        add(build_geocode_fallback_query(node))

    city_key = _norm_key(node.city or "")
    for alias in _RESORT_CITY_ALIASES.get(city_key, ()):
        add(alias)
    latin_key = _norm_key(node.city or "").replace('"', "")
    if latin_key in _CITY_ALIASES:
        add(_CITY_ALIASES[latin_key])

    return out


class CacheOnlyGeocoder:
    """Resolve strictly from an on-disk cache (no network)."""

    def __init__(self, cache: JsonGeocodeCache):
        self.cache = cache

    def geocode(self, query: str) -> GeocodeResult | None:
        return self.cache.get(query)


def geocode_node(
    node: CanonicalNode,
    geocoder: Geocoder,
    *,
    use_city_fallback: bool = True,
) -> tuple[CanonicalNode, bool]:
    """Resolve coordinates for a single node. Returns ``(node, resolved)``."""
    if node.has_coordinates:
        return node, True

    queries = build_geocode_queries(node, use_city_fallback=use_city_fallback)
    cache = getattr(geocoder, "cache", None)

    for query in queries:
        hit = cache.get(query) if cache is not None else None
        if hit is not None:
            return (
                replace(
                    node,
                    latitude=hit.latitude,
                    longitude=hit.longitude,
                    quality_score=min(1.0, node.quality_score * hit.confidence),
                ),
                True,
            )

    for query in queries:
        if cache is not None and cache.get(query) is not None:
            continue
        result = geocoder.geocode(query)
        if result is not None:
            return (
                replace(
                    node,
                    latitude=result.latitude,
                    longitude=result.longitude,
                    quality_score=min(1.0, node.quality_score * result.confidence),
                ),
                True,
            )
    return node, False


def resolve_nodes_from_cache(
    nodes: list[CanonicalNode],
    cache_path: str | Path,
    *,
    use_city_fallback: bool = True,
) -> tuple[list[CanonicalNode], list[CanonicalNode]]:
    """Apply cache hits in memory only. Returns ``(all_nodes, still_pending)``."""
    cache = JsonGeocodeCache(cache_path)
    geocoder = CacheOnlyGeocoder(cache)
    out: list[CanonicalNode] = []
    pending: list[CanonicalNode] = []
    for node in nodes:
        if node.has_coordinates:
            out.append(node)
            continue
        resolved_node, ok = geocode_node(
            node, geocoder, use_city_fallback=use_city_fallback
        )
        out.append(resolved_node)
        if not ok:
            pending.append(resolved_node)
    return out, pending


def geocode_nodes(
    nodes: list[CanonicalNode],
    geocoder: Geocoder,
    *,
    max_new_requests: int | None = None,
    use_city_fallback: bool = True,
    cache_for_offline: JsonGeocodeCache | None = None,
) -> tuple[list[CanonicalNode], list[str]]:
    out: list[CanonicalNode] = []
    unresolved: list[str] = []
    used = 0
    offline = (
        CacheOnlyGeocoder(cache_for_offline)
        if cache_for_offline is not None
        else None
    )
    for node in nodes:
        if node.has_coordinates:
            out.append(node)
            continue
        budget_exhausted = max_new_requests is not None and used >= max_new_requests
        if budget_exhausted and offline is not None:
            resolved_node, resolved = geocode_node(
                node, offline, use_city_fallback=use_city_fallback
            )
            out.append(resolved_node)
            if not resolved:
                unresolved.append(node.node_id)
            continue
        if budget_exhausted:
            unresolved.append(node.node_id)
            out.append(node)
            continue
        resolved_node, resolved = geocode_node(
            node, geocoder, use_city_fallback=use_city_fallback
        )
        used += 1
        out.append(resolved_node)
        if not resolved:
            unresolved.append(node.node_id)
    return out, unresolved

