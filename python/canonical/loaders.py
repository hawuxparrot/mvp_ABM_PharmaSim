from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


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

