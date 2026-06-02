from __future__ import annotations

from collections import defaultdict

import numpy as np

from policy.models import OrgType, Scenario
from policy.transactions import TransactionIntent, TransactionPlan, TransactionStage


def _build_location_org_type_lookup(scenario: Scenario) -> dict[str, OrgType]:
    org_type_by_org = {o.ext_id: o.org_type for o in scenario.organizations}
    return {
        loc.ext_id: org_type_by_org[loc.org_ext_id]
        for loc in scenario.locations
        if loc.org_ext_id in org_type_by_org
    }


def _build_location_market_lookup(scenario: Scenario) -> dict[str, str]:
    return {loc.ext_id: loc.market_code for loc in scenario.locations}


def _candidate_supply_edges(
    scenario: Scenario,
) -> list[tuple[str, str, float]]:
    loc_type = _build_location_org_type_lookup(scenario)
    candidates: list[tuple[str, str, float]] = []
    for edge in scenario.location_edges:
        src_t = loc_type.get(edge.src_location_ext_id)
        dst_t = loc_type.get(edge.dst_location_ext_id)
        if src_t in {OrgType.OBP, OrgType.WHOLESALER} and dst_t in {
            OrgType.WHOLESALER,
            OrgType.LOCAL_ORG,
        }:
            candidates.append(
                (
                    edge.src_location_ext_id,
                    edge.dst_location_ext_id,
                    float(edge.cost),
                )
            )
    return candidates


def _fallback_edges(scenario: Scenario) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for edge in scenario.location_edges:
        out.append(
            (
                edge.src_location_ext_id,
                edge.dst_location_ext_id,
                float(edge.cost),
            )
        )
    return out


def build_synthetic_transaction_plan(
    scenario: Scenario,
    *,
    horizon_ticks: int = 30,
    order_lambda_per_edge: float = 0.6,
    max_units_per_order: int = 40,
    seed: int | None = None,
) -> TransactionPlan:
    """
    Build a simple synthetic transaction plan aligned with the scenario graph.

    This plan is independent from the native engine tick loop and acts as an
    experiment-side control stream for fraud/anomaly workflows.
    """
    if horizon_ticks <= 0:
        raise ValueError("horizon_ticks must be > 0")
    if max_units_per_order <= 0:
        raise ValueError("max_units_per_order must be > 0")

    rng = np.random.default_rng(seed if seed is not None else scenario.seed)
    candidates = _candidate_supply_edges(scenario)
    if not candidates:
        candidates = _fallback_edges(scenario)
    if not candidates:
        # Final fallback: generate light traffic between random location pairs so
        # downstream fraud experimentation remains usable even on sparse scenarios.
        locs = [loc.ext_id for loc in scenario.locations]
        if len(locs) < 2:
            return TransactionPlan(intents=[])
        for i in range(min(8, len(locs) - 1)):
            candidates.append((locs[i], locs[i + 1], 1.0))

    market_by_loc = _build_location_market_lookup(scenario)
    product_id = scenario.products[0].ext_id if scenario.products else "prod_unknown"

    intents: list[TransactionIntent] = []
    tx_counter = 0
    for tick in range(horizon_ticks):
        for src_loc, dst_loc, edge_cost in candidates:
            n_orders = int(rng.poisson(order_lambda_per_edge))
            for _ in range(n_orders):
                tx_counter += 1
                units = int(rng.integers(1, max_units_per_order + 1))
                src_market = market_by_loc.get(src_loc, "UNK")
                dst_market = market_by_loc.get(dst_loc, "UNK")

                # Three-stage lifecycle with deterministic IDs derived from one root.
                base = f"tx_{tick:04d}_{tx_counter:08d}"
                intents.append(
                    TransactionIntent(
                        tx_id=f"{base}_order",
                        tick=tick,
                        stage=TransactionStage.ORDER_CREATED,
                        src_location_ext_id=src_loc,
                        dst_location_ext_id=dst_loc,
                        src_market_code=src_market,
                        dst_market_code=dst_market,
                        product_ext_id=product_id,
                        units=units,
                        route_cost_hint=edge_cost,
                        metadata={"lifecycle_root": base},
                    )
                )
                intents.append(
                    TransactionIntent(
                        tx_id=f"{base}_dispatch",
                        tick=tick,
                        stage=TransactionStage.SHIPMENT_DISPATCHED,
                        src_location_ext_id=src_loc,
                        dst_location_ext_id=dst_loc,
                        src_market_code=src_market,
                        dst_market_code=dst_market,
                        product_ext_id=product_id,
                        units=units,
                        route_cost_hint=edge_cost,
                        metadata={"lifecycle_root": base},
                    )
                )
                intents.append(
                    TransactionIntent(
                        tx_id=f"{base}_delivery",
                        tick=tick + 1,
                        stage=TransactionStage.SHIPMENT_DELIVERED,
                        src_location_ext_id=src_loc,
                        dst_location_ext_id=dst_loc,
                        src_market_code=src_market,
                        dst_market_code=dst_market,
                        product_ext_id=product_id,
                        units=units,
                        route_cost_hint=edge_cost,
                        metadata={"lifecycle_root": base},
                    )
                )
    intents.sort(key=lambda x: (x.tick, x.tx_id))
    return TransactionPlan(intents=intents)


def plan_stage_counts(plan: TransactionPlan) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for intent in plan.intents:
        counts[intent.stage.value] += 1
    return dict(counts)

