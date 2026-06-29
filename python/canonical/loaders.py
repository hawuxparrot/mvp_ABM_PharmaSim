from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from policy.models import Product, ProductCode, ProductCodeScheme


def load_bg_registry_rows(path: str | Path) -> list[dict[str, Any]]:
    src = Path(path)
    raw = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("BG registry source must be a JSON list")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"BG registry row {i} is not an object")
        rows.append(item)
    return rows


def _parse_spor_row(row: list[str], field_count: int) -> list[str]:
    if len(row) == field_count:
        return row
    if len(row) == 1:
        reparsed = next(csv.reader([row[0]]))
        if len(reparsed) == field_count:
            return reparsed
    raise ValueError(
        f"SPOR row has unexpected width {len(row)} (expected {field_count})"
    )


_PRODUCT_REQUIRED_COLUMNS = ("product_id", "name", "atc_code")


def load_product_rows(path: str | Path) -> list[dict[str, str]]:
    """Load the product catalog CSV into raw row dicts (all columns preserved)."""
    src = Path(path)
    rows: list[dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _PRODUCT_REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"products CSV missing required columns: {missing}")
        for i, record in enumerate(reader):
            pid = (record.get("product_id") or "").strip()
            if not pid:
                raise ValueError(f"products CSV row {i} has empty product_id")
            rows.append({k: (v or "").strip() for k, v in record.items()})
    return rows


def _product_codes(row: dict[str, str]) -> tuple[ProductCode, ...]:
    codes: list[ProductCode] = []
    gtin = row.get("gtin", "")
    ppn = row.get("ppn", "")
    if gtin:
        codes.append(ProductCode(scheme=ProductCodeScheme.GTIN, value=gtin, is_primary=True))
    if ppn:
        codes.append(
            ProductCode(
                scheme=ProductCodeScheme.PPN, value=ppn, is_primary=not bool(gtin)
            )
        )
    if not codes:
        # Synthesize a placeholder GTIN so every product has at least one code.
        codes.append(
            ProductCode(
                scheme=ProductCodeScheme.GTIN,
                value=row["product_id"],
                is_primary=True,
            )
        )
    return tuple(codes)


def canonicalize_products(rows: list[dict[str, str]]) -> list[Product]:
    """Turn product-catalog rows into :class:`Product` objects carrying their ATC code."""
    products: list[Product] = []
    for row in rows:
        pack_size_raw = row.get("pack_size", "")
        try:
            pack_size = int(pack_size_raw) if pack_size_raw else None
        except ValueError:
            pack_size = None
        atc = (row.get("atc_code") or "").strip().upper() or None
        products.append(
            Product(
                ext_id=row["product_id"],
                codes=_product_codes(row),
                atc_code=atc,
                name=row.get("name") or None,
                pack_size=pack_size,
            )
        )
    return products


def product_intended_markets(rows: list[dict[str, str]]) -> dict[str, tuple[str, ...]]:
    """Map product_id -> intended market codes parsed from the catalog CSV."""
    out: dict[str, tuple[str, ...]] = {}
    for row in rows:
        raw = row.get("intended_markets", "") or "BG"
        markets = tuple(
            m.strip().upper()
            for m in raw.replace(";", "|").replace(",", "|").split("|")
            if m.strip()
        )
        out[row["product_id"]] = markets or ("BG",)
    return out


def product_substitution_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    """Map product_id -> manual ``substitution_group_id`` for rows that declare one.

    Optional column; products without it fall back to ATC-based substitutability.
    """
    out: dict[str, str] = {}
    for row in rows:
        gid = (row.get("substitution_group_id") or "").strip()
        if gid:
            out[row["product_id"]] = gid
    return out


def load_spor_rows(
    path: str | Path,
    *,
    country_code: str | None = None,
) -> list[dict[str, str]]:
    src = Path(path)
    rows: list[dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        if headers and headers[0].startswith("\ufeff"):
            headers[0] = headers[0].lstrip("\ufeff")
        field_count = len(headers)
        for row in reader:
            parsed = _parse_spor_row(row, field_count)
            record = dict(zip(headers, parsed))
            if country_code is not None:
                cc = (record.get("Address Country Code") or "").strip()
                if cc != country_code:
                    continue
            rows.append(record)
    return rows

