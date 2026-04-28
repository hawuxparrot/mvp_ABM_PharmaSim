"""
Large synthetic scenarios: multiple markets, many locations, sparse directed graphs.

**Scaling note: A naive Erdős–Rényi model with edge probability ``p`` on ``n`` nodes has
expected directed edges ``p * n * (n-1)``. For ``n=10_000`` and ``p=0.1`` that is about
**10 million** directed edges per market (40M across four markets), which is usually
impractical to build in Python and heavy in RAM. This module therefore:

- defaults to a **fixed out-degree** model (sparse, predictable edge count);
- supports ``erdos_renyi`` with an explicit ``max_directed_edges_per_market`` cap (and a
  warning if the cap bites).

Example::

    from policy.scenarios_large import multi_market_sparse_scenario
    from compiler.compile import compile_scenario

    s = multi_market_sparse_scenario(
        locations_per_market=10_000,
        graph_model=\"out_degree\",
        out_degree=25,
    )
    inp = compile_scenario(s)
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import Literal

import numpy as np

from policy.models import (
    Batch,
    Location,
    LocationEdge,
    LocationBehavior,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario,
)

GraphModel = Literal["out_degree", "erdos_renyi"]


def _loc_id(market: str, idx: int) -> str:
    return f"loc_{market}_{idx}"


def _sample_directed_edges_unique(
    n: int,
    m_target: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Up to m_target distinct directed pairs (i,j), i != j, uniform over allowed pairs."""
    if n < 2 or m_target <= 0:
        return []
    total_pairs = n * (n - 1)
    m = min(m_target, total_pairs)
    seen: set[tuple[int, int]] = set()
    batch = 200_000
    while len(seen) < m:
        a = rng.integers(0, n, size=batch, dtype=np.int32)
        b = rng.integers(0, n, size=batch, dtype=np.int32)
        for i in range(batch):
            ia = int(a[i])
            ib = int(b[i])
            if ia == ib:
                continue
            seen.add((ia, ib))
            if len(seen) >= m:
                break
    return sorted(seen)[:m]


def _edges_out_degree(
    n: int,
    out_degree: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    d = min(out_degree, max(0, n - 1))
    if d == 0:
        return []
    edges: set[tuple[int, int]] = set()
    for i in range(n):
        # Sample d distinct targets != i
        choices = np.arange(n, dtype=np.int32)
        choices = np.delete(choices, i)
        rng.shuffle(choices)
        for j in choices[:d]:
            edges.add((i, int(j)))
    return sorted(edges)


def _edges_erdos_renyi_capped(
    n: int,
    p: float,
    rng: np.random.Generator,
    max_edges: int,
) -> tuple[list[tuple[int, int]], int, float]:
    """
    Directed ER with binomial edge count, capped at ``max_edges``.

    Returns (edges, target_m_before_cap, effective_p_used_for_message).
    """
    total_pairs = n * (n - 1)
    if p <= 0.0 or n < 2:
        return [], 0, p
    m_draw = int(rng.binomial(total_pairs, min(p, 1.0)))
    capped = False
    if m_draw > max_edges:
        capped = True
        m_draw = max_edges
    edges = _sample_directed_edges_unique(n, m_draw, rng)
    p_eff = (len(edges) / total_pairs) if total_pairs > 0 else 0.0
    if capped:
        warnings.warn(
            f"erdos_renyi: capped directed edges at {max_edges} per market "
            f"(n={n}, requested p={p} would target ~{p * total_pairs:.0f} edges). "
            f"Effective p ≈ {p_eff:.6g}. "
            f"Raise max_directed_edges_per_market or lower p / n.",
            stacklevel=2,
        )
    return edges, m_draw, p_eff


def multi_market_sparse_scenario_precomputed(
    *,
    market_codes: tuple[str, ...] = ("DE", "FR", "IT", "UK"),
    locations_per_market: int = 100_000,
    seed: int = 42,
    graph_model: GraphModel = "out_degree",
    out_degree: int = 20,
    er_edge_probability: float = 0.2,
    max_directed_edges_per_market: int = 1_000_000,
    wholesaler_site_count_per_market: int = 100,
    local_org_pool_per_market: int = 200,
    packs_per_market: int = 2_000_000,
    behavior_location_fraction: float = 0.0,
    behavior_verify_prob: float = 0.1,
    behavior_decommission_prob: float = 0.02,
    behavior_reactivate_prob: float = 0.005,
) -> tuple[Scenario, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same scenario as :func:`multi_market_sparse_scenario`, but **without** materializing
    ``list[LocationEdge]``. Returns a scenario with ``location_edges=()`` and columnar
    edge arrays (global dense location indices) suitable for
    :func:`compiler.compile.compile_scenario_with_precomputed_edges`.

    Intra-market edge pairs are **sorted** by ``(src_local, dst_local)`` for deterministic
    ordering (same as the list-based builder).
    """
    if locations_per_market < 3:
        raise ValueError("locations_per_market must be at least 3 (OBP + wholesaler + local)")
    wh = min(wholesaler_site_count_per_market, locations_per_market - 2)
    if wh < 1:
        raise ValueError("need at least one wholesaler site; increase locations_per_market")

    rng = np.random.default_rng(seed)
    all_markets = list(market_codes)
    intended_all = tuple(sorted(all_markets))

    orgs: list[Organization] = []
    locs: list[Location] = []
    behavior: dict[str, LocationBehavior] = {}

    def org_obp(mc: str) -> str:
        return f"obp_{mc}"

    def org_wh(mc: str, k: int) -> str:
        return f"wh_{mc}_{k}"

    def org_lo(mc: str, k: int) -> str:
        return f"lo_{mc}_{k}"

    for mc in market_codes:
        orgs.append(Organization(ext_id=org_obp(mc), org_type=OrgType.OBP))
        for k in range(wh):
            orgs.append(Organization(ext_id=org_wh(mc, k), org_type=OrgType.WHOLESALER))
        for k in range(local_org_pool_per_market):
            orgs.append(Organization(ext_id=org_lo(mc, k), org_type=OrgType.LOCAL_ORG))

        for i in range(locations_per_market):
            if i == 0:
                oid = org_obp(mc)
            elif i <= wh:
                oid = org_wh(mc, i - 1)
            else:
                pool_ix = (i - wh - 1) % local_org_pool_per_market
                oid = org_lo(mc, pool_ix)
            locs.append(
                Location(
                    ext_id=_loc_id(mc, i),
                    org_ext_id=oid,
                    market_code=mc,
                    postal_code=f"p{i % 100000:05d}",
                )
            )

    es: list[int] = []
    ed: list[int] = []
    ec: list[float] = []
    ek: list[int] = []
    global_cursor = 0
    for mc in market_codes:
        n = locations_per_market
        base = global_cursor
        global_cursor += n

        if graph_model == "out_degree":
            local_edges = _edges_out_degree(n, out_degree, rng)
        elif graph_model == "erdos_renyi":
            local_edges, _, _ = _edges_erdos_renyi_capped(
                n, er_edge_probability, rng, max_directed_edges_per_market
            )
        else:
            raise ValueError(f"unknown graph_model: {graph_model!r}")

        for i, j in local_edges:
            es.append(base + i)
            ed.append(base + j)
            ec.append(float(rng.uniform(0.5, 3.0)))
            ek.append(int(rng.integers(100, 2001)))

        if behavior_location_fraction > 0.0:
            local_indices = np.arange(wh + 1, n, dtype=np.int32)
            if local_indices.size > 0:
                mask = rng.random(local_indices.size) < behavior_location_fraction
                chosen = local_indices[mask]
                for li in chosen:
                    lid = _loc_id(mc, int(li))
                    behavior[lid] = LocationBehavior(
                        verify_prob=behavior_verify_prob,
                        decomission_prob=behavior_decommission_prob,
                        reactivate_prob=behavior_reactivate_prob,
                    )

    product = Product(
        ext_id="prod_synthetic",
        codes=(
            ProductCode(
                scheme=ProductCodeScheme.GTIN,
                value="01234567891011",
                is_primary=True,
            ),
        ),
    )
    products = [product]

    batches: list[Batch] = []
    packs: list[Pack] = []
    for mc in market_codes:
        bid = f"batch_{mc}"
        batches.append(
            Batch(
                ext_id=bid,
                product_ext_id=product.ext_id,
                manufacturer_org_ext_id=org_obp(mc),
                intended_markets=intended_all,
            )
        )
        obp_loc = _loc_id(mc, 0)
        for k in range(packs_per_market):
            packs.append(
                Pack(
                    ext_id=f"pack_{mc}_{k}",
                    product_ext_id=product.ext_id,
                    batch_ext_id=bid,
                    serial=f"SN_{mc}_{k}",
                    initial_market_code=mc,
                    initial_location_ext_id=obp_loc,
                    initial_state=PackState.UPLOADED,
                )
            )

    edge_src = np.asarray(es, dtype=np.uint32)
    edge_dst = np.asarray(ed, dtype=np.uint32)
    edge_cost = np.asarray(ec, dtype=np.float32)
    edge_capacity = np.asarray(ek, dtype=np.uint32)

    scenario = Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=[],
        behavior_by_location=behavior,
        seed=seed,
    )
    return scenario, edge_src, edge_dst, edge_cost, edge_capacity


def multi_market_sparse_scenario(
    *,
    market_codes: tuple[str, ...] = ("DE", "FR", "IT", "UK"),
    locations_per_market: int = 50_000,
    seed: int = 42,
    graph_model: GraphModel = "out_degree",
    out_degree: int = 20,
    er_edge_probability: float = 0.1,
    max_directed_edges_per_market: int = 100_000,
    wholesaler_site_count_per_market: int = 1000,
    local_org_pool_per_market: int = 10_000,
    packs_per_market: int = 1_000_000,
    behavior_location_fraction: float = 1.0,
    behavior_verify_prob: float = 0.1,
    behavior_decommission_prob: float = 0.02,
    behavior_reactivate_prob: float = 0.005,
) -> Scenario:
    """
    Build a multi-market scenario with a sparse **within-market** directed location graph.

    Per market:

    - Index ``0``: OBP manufacturing site (one ``Organization`` of type OBP).
    - Indices ``1 .. wholesaler_site_count``: one location each for distinct wholesalers.
    - Remaining indices: local org (pharmacy/hospital) sites sharing a pool of
      ``local_org_pool_per_market`` organizations.

    **No cross-market edges** are generated (export-style flows would be extra edges you
    add later).

    :param graph_model: ``out_degree`` (default) or ``erdos_renyi``.
    :param er_edge_probability: used only if ``graph_model == \"erdos_renyi\"``; subject
        to ``max_directed_edges_per_market``.
    :param behavior_location_fraction: fraction of **local** sites (not OBP/wholesaler)
        that get stochastic ``LocationBehavior`` entries (0 keeps behavior empty for speed).
    """
    if locations_per_market < 3:
        raise ValueError("locations_per_market must be at least 3 (OBP + wholesaler + local)")
    wh = min(wholesaler_site_count_per_market, locations_per_market - 2)
    if wh < 1:
        raise ValueError("need at least one wholesaler site; increase locations_per_market")

    rng = np.random.default_rng(seed)
    all_markets = list(market_codes)
    intended_all = tuple(sorted(all_markets))

    orgs: list[Organization] = []
    locs: list[Location] = []
    edges: list[LocationEdge] = []
    behavior: dict[str, LocationBehavior] = {}

    def org_obp(mc: str) -> str:
        return f"obp_{mc}"

    def org_wh(mc: str, k: int) -> str:
        return f"wh_{mc}_{k}"

    def org_lo(mc: str, k: int) -> str:
        return f"lo_{mc}_{k}"

    for mc in market_codes:
        orgs.append(Organization(ext_id=org_obp(mc), org_type=OrgType.OBP))
        for k in range(wh):
            orgs.append(Organization(ext_id=org_wh(mc, k), org_type=OrgType.WHOLESALER))
        for k in range(local_org_pool_per_market):
            orgs.append(Organization(ext_id=org_lo(mc, k), org_type=OrgType.LOCAL_ORG))

        for i in range(locations_per_market):
            if i == 0:
                oid = org_obp(mc)
            elif i <= wh:
                oid = org_wh(mc, i - 1)
            else:
                pool_ix = (i - wh - 1) % local_org_pool_per_market
                oid = org_lo(mc, pool_ix)
            locs.append(
                Location(
                    ext_id=_loc_id(mc, i),
                    org_ext_id=oid,
                    market_code=mc,
                    postal_code=f"p{i % 100000:05d}",
                )
            )

    # Intra-market edges: local indices 0..n-1 map to consecutive global location rows
    # in the order markets were appended (same order as ``market_codes``).
    global_cursor = 0
    for mc in market_codes:
        n = locations_per_market
        base = global_cursor
        global_cursor += n

        if graph_model == "out_degree":
            local_edges = _edges_out_degree(n, out_degree, rng)
        elif graph_model == "erdos_renyi":
            local_edges, _, _ = _edges_erdos_renyi_capped(
                n, er_edge_probability, rng, max_directed_edges_per_market
            )
        else:
            raise ValueError(f"unknown graph_model: {graph_model!r}")

        for i, j in local_edges:
            src = _loc_id(mc, i)
            dst = _loc_id(mc, j)
            c = float(rng.uniform(0.5, 3.0))
            cap = int(rng.integers(100, 2001))
            edges.append(
                LocationEdge(src_location_ext_id=src, dst_location_ext_id=dst, cost=c, capacity=cap)
            )

        # Optional behavior on a random subset of local-only sites (index > wh)
        if behavior_location_fraction > 0.0:
            local_indices = np.arange(wh + 1, n, dtype=np.int32)
            if local_indices.size > 0:
                mask = rng.random(local_indices.size) < behavior_location_fraction
                chosen = local_indices[mask]
                for li in chosen:
                    lid = _loc_id(mc, int(li))
                    behavior[lid] = LocationBehavior(
                        verify_prob=behavior_verify_prob,
                        decomission_prob=behavior_decommission_prob,
                        reactivate_prob=behavior_reactivate_prob,
                    )

    product = Product(
        ext_id="prod_synthetic",
        codes=(
            ProductCode(
                scheme=ProductCodeScheme.GTIN,
                value="01234567891011",
                is_primary=True,
            ),
        ),
    )
    products = [product]

    batches: list[Batch] = []
    packs: list[Pack] = []
    for mc in market_codes:
        bid = f"batch_{mc}"
        batches.append(
            Batch(
                ext_id=bid,
                product_ext_id=product.ext_id,
                manufacturer_org_ext_id=org_obp(mc),
                intended_markets=intended_all,
            )
        )
        obp_loc = _loc_id(mc, 0)
        for k in range(packs_per_market):
            packs.append(
                Pack(
                    ext_id=f"pack_{mc}_{k}",
                    product_ext_id=product.ext_id,
                    batch_ext_id=bid,
                    serial=f"SN_{mc}_{k}",
                    initial_market_code=mc,
                    initial_location_ext_id=obp_loc,
                    initial_state=PackState.UPLOADED,
                )
            )

    return Scenario(
        organizations=orgs,
        locations=locs,
        products=products,
        batches=batches,
        packs=packs,
        location_edges=edges,
        behavior_by_location=behavior,
        seed=seed,
    )


def iter_location_slices_by_market(
    market_codes: tuple[str, ...],
    locations_per_market: int,
) -> Iterator[tuple[str, slice]]:
    """Yield ``(market_code, slice)`` into a locations list built like :func:`multi_market_sparse_scenario`."""
    start = 0
    for mc in market_codes:
        yield mc, slice(start, start + locations_per_market)
        start += locations_per_market
