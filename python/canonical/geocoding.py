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


def geocode_nodes(
    nodes: list[CanonicalNode],
    geocoder: Geocoder,
    *,
    max_new_requests: int | None = None,
) -> tuple[list[CanonicalNode], list[str]]:
    out: list[CanonicalNode] = []
    unresolved: list[str] = []
    used = 0
    for node in nodes:
        if node.has_coordinates:
            out.append(node)
            continue
        if max_new_requests is not None and used >= max_new_requests:
            unresolved.append(node.node_id)
            out.append(node)
            continue
        query = build_geocode_query(node)
        result = geocoder.geocode(query)
        used += 1
        if result is None:
            unresolved.append(node.node_id)
            out.append(node)
            continue
        out.append(
            replace(
                node,
                latitude=result.latitude,
                longitude=result.longitude,
                quality_score=min(1.0, node.quality_score * result.confidence),
            )
        )
    return out, unresolved

