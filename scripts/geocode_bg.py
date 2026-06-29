#!/usr/bin/env python
"""Resumable batch geocoder for the Bulgaria canonical node set.

Populates ``data/geocode_cache_bg.json`` with OSM/Nominatim coordinates for every
active canonical node (BG-registry pharmacies + SPOR locations). The cache is written
after every successful lookup, so the job is fully resumable: re-running it only issues
network requests for nodes still missing coordinates.

Usage:
    uv run python scripts/geocode_bg.py [--max-new N] [--limit N]

Progress goes to stderr by default (``tqdm`` bar with ETA). When backgrounding, either
leave stderr on the terminal or follow the log::

    uv run python scripts/geocode_bg.py 2>> out_logs/geocode_bg.log
    tail -f out_logs/geocode_bg.log

Nominatim usage policy requires <= 1 request/second; the default provider already
pauses between requests. A full cold run therefore takes roughly one request per
unresolved node/city and should be left running in the background once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Allow running as a plain script (python scripts/geocode_bg.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from canonical.geocoding import (  # noqa: E402
    CachedGeocoder,
    JsonGeocodeCache,
    NominatimGeocoder,
    geocode_node,
    resolve_nodes_from_cache,
)
from canonical.loaders import load_bg_registry_rows, load_spor_rows  # noqa: E402
from canonical.normalize import (  # noqa: E402
    canonicalize_bg_registry,
    canonicalize_spor,
    merge_nodes,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bg-registry", default=str(repo / "data" / "data.json"))
    p.add_argument("--spor", default=str(repo / "data" / "spor_locations.csv"))
    p.add_argument("--cache", default=str(repo / "data" / "geocode_cache_bg.json"))
    p.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Cap on new network lookups this run (omit to resolve everything).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only consider the first N nodes (debugging).",
    )
    p.add_argument(
        "--no-city-fallback",
        action="store_true",
        help="Disable the city-centroid fallback query.",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause between live requests (Nominatim policy: >= 1).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar (plain log lines only).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    bg_nodes = canonicalize_bg_registry(load_bg_registry_rows(args.bg_registry))
    spor_nodes = canonicalize_spor(load_spor_rows(args.spor, country_code="BG"))
    nodes = [n for n in merge_nodes(bg_nodes, spor_nodes) if n.is_active]
    if args.limit is not None:
        nodes = nodes[: args.limit]

    geocoder = CachedGeocoder(
        provider=NominatimGeocoder(pause_seconds=args.pause),
        cache=JsonGeocodeCache(args.cache),
    )

    total = len(nodes)
    already = sum(1 for n in nodes if n.has_coordinates)
    _all_nodes, pending_nodes = resolve_nodes_from_cache(
        nodes,
        args.cache,
        use_city_fallback=not args.no_city_fallback,
    )
    pending = len(pending_nodes)
    to_process = (
        pending_nodes
        if args.max_new is None
        else pending_nodes[: args.max_new]
    )

    print(
        f"nodes={total} with_coords={already} pending={pending} "
        f"this_run={len(to_process)} cache={args.cache}",
        flush=True,
    )

    if not to_process:
        print("nothing to geocode", flush=True)
        return 0

    # stderr so progress survives ``stdout`` redirection; disable when not a TTY unless forced.
    show_progress = not args.no_progress
    bar = tqdm(
        to_process,
        desc="geocoding",
        unit="req",
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )

    resolved = 0
    failed = 0
    started = time.time()
    for node in bar:
        _updated, ok = geocode_node(
            node, geocoder, use_city_fallback=not args.no_city_fallback
        )
        if ok:
            resolved += 1
        else:
            failed += 1
        elapsed = max(1e-6, time.time() - started)
        bar.set_postfix(
            ok=resolved,
            fail=failed,
            rate=f"{bar.n / elapsed:.2f}/s",
            refresh=False,
        )

    bar.close()
    lookups = resolved + failed
    print(
        f"done: lookups={lookups} resolved={resolved} failed={failed} "
        f"cache_written={args.cache}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
