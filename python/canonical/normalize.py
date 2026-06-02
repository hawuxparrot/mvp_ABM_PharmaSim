from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .models import ActorRole, CanonicalNode


def _pick_multilingual_variant(value: str | None) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    if "¦" in raw:
        return raw.split("¦", maxsplit=1)[0].strip()
    return raw


def normalize_text(value: str | None) -> str:
    picked = _pick_multilingual_variant(value)
    folded = unicodedata.normalize("NFKC", picked)
    compact = re.sub(r"\s+", " ", folded).strip()
    return compact


def normalize_key(value: str | None) -> str:
    text = normalize_text(value).lower()
    return "".join(ch for ch in text if ch.isalnum() or ch == " ")


def _parse_float_pair(gps: str | None) -> tuple[float | None, float | None]:
    if not gps:
        return (None, None)
    parts = [p.strip() for p in gps.split(",")]
    if len(parts) != 2:
        return (None, None)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (None, None)


def _role_from_bg_registry(a_type: Any) -> ActorRole:
    try:
        type_int = int(a_type)
    except (TypeError, ValueError):
        return ActorRole.UNKNOWN
    # The source coding is not documented in-repo; keep conservative mappings.
    if type_int == 2:
        return ActorRole.PHARMACY
    if type_int == 3:
        return ActorRole.HOSPITAL
    if type_int == 1:
        return ActorRole.WHOLESALER
    return ActorRole.UNKNOWN


def _role_from_spor_category(category_display: str) -> ActorRole:
    c = normalize_key(category_display)
    if "hospital" in c or "clinic" in c:
        return ActorRole.HOSPITAL
    if "regulatory authority" in c or "ethics committee" in c:
        return ActorRole.REGULATOR
    if "industry" in c or "pharmaceutical company" in c:
        return ActorRole.WHOLESALER
    return ActorRole.UNKNOWN


def canonicalize_bg_registry(rows: Iterable[dict[str, Any]]) -> list[CanonicalNode]:
    out: list[CanonicalNode] = []
    for row in rows:
        is_active = bool(row.get("is_active"))
        if not is_active:
            continue
        apteka_n = normalize_text(str(row.get("apteka_n", "")))
        if not apteka_n:
            continue
        node = CanonicalNode(
            node_id=f"bg-reg-{apteka_n}",
            name=normalize_text(row.get("c_name")) or normalize_text(row.get("full_n")),
            role=_role_from_bg_registry(row.get("a_type")),
            market_code="BG",
            city=normalize_text(row.get("a_town")),
            address=normalize_text(row.get("a_address")),
            postal_code="",
            is_active=True,
            latitude=None,
            longitude=None,
            source_ids={"bg_registry.apteka_n": apteka_n},
            quality_score=1.0,
        )
        out.append(node)
    return out


def canonicalize_spor(rows: Iterable[dict[str, str]]) -> list[CanonicalNode]:
    out: list[CanonicalNode] = []
    for row in rows:
        loc_id = normalize_text(row.get("Location ID"))
        if not loc_id:
            continue
        lat, lon = _parse_float_pair(row.get("Address GPS Location"))
        node = CanonicalNode(
            node_id=f"spor-{loc_id}",
            name=normalize_text(row.get("Name")),
            role=_role_from_spor_category(
                normalize_text(row.get("Category Classification Category  Display Name"))
            ),
            market_code=normalize_text(row.get("Address Country Code")) or "BG",
            city=normalize_text(row.get("Address City")),
            address=normalize_text(row.get("Address Line 1")),
            postal_code=normalize_text(row.get("Address Postal Code")),
            is_active=normalize_key(row.get("Status")) == "active",
            latitude=lat,
            longitude=lon,
            source_ids={"spor.location_id": loc_id},
            quality_score=0.9,
        )
        out.append(node)
    return out


def merge_nodes(primary: list[CanonicalNode], secondary: list[CanonicalNode]) -> list[CanonicalNode]:
    """
    Merge nodes by coarse geographic key so source-specific IDs can co-exist.

    Primary nodes win for role/name stability; secondary rows enrich coordinates.
    """
    by_key: dict[str, CanonicalNode] = {}
    key_to_id: dict[str, str] = {}

    def merge_key(node: CanonicalNode) -> str:
        return "|".join(
            [
                normalize_key(node.city),
                normalize_key(node.address),
                normalize_key(node.postal_code),
            ]
        )

    for node in primary:
        key = merge_key(node)
        by_key[key] = node
        key_to_id[key] = node.node_id

    for node in secondary:
        key = merge_key(node)
        if key in by_key:
            incumbent = by_key[key]
            lat = incumbent.latitude if incumbent.latitude is not None else node.latitude
            lon = incumbent.longitude if incumbent.longitude is not None else node.longitude
            source_ids = dict(incumbent.source_ids)
            source_ids.update(node.source_ids)
            by_key[key] = replace(
                incumbent,
                latitude=lat,
                longitude=lon,
                source_ids=source_ids,
                quality_score=max(incumbent.quality_score, node.quality_score),
            )
        else:
            by_key[key] = node
            key_to_id[key] = node.node_id

    return list(by_key.values())

