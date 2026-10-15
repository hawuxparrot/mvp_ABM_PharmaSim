"""The EMVO/NMVS data request, mapped from meeting numbers to engine parameters.

Each row ties a figure or gap the supervisor raised to (a) what the simulation does
today, (b) the concrete ``EngineInput`` parameter the requested data would calibrate,
and (c) the anonymized aggregate we would ask the EMVO/NMVO for. The same rows drive
the dashboard's data-request tab and ``docs/emvo_data_request.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataRequestItem:
    priority: str
    topic: str
    meeting_signal: str
    simulation_today: str
    engine_parameter: str
    requested_aggregate: str


EMVO_DATA_REQUEST: tuple[DataRequestItem, ...] = (
    DataRequestItem(
        priority="1",
        topic="Verify / decommission rates by site type",
        meeting_signal="Dispense scans near-mandatory; alert rate target 0.05%",
        simulation_today="verify_prob=0.12, decommission_prob=0.03 for every local org",
        engine_parameter="location_verify_prob, location_decommission_prob",
        requested_aggregate="Verifications & decommissions per site per day, split "
        "community pharmacy vs hospital (p50/p95)",
    ),
    DataRequestItem(
        priority="1",
        topic="Alert incidence and outcomes",
        meeting_signal="BG ~3000 alerts/week, most errors, ~70% from one vendor",
        simulation_today="Alerts emitted by EMVS rules; no real incidence baseline",
        engine_parameter="detector thresholds, candidate rates",
        requested_aggregate="Alert incidence per 10^4 scans by code, and "
        "alert -> investigation outcome (real fraud vs error)",
    ),
    DataRequestItem(
        priority="1",
        topic="Licensed supply network",
        meeting_signal="Realistic routes / graph of who supplies whom",
        simulation_today="k-NN haversine proxy edges (routing.py), not real links",
        engine_parameter="location_edges (src/dst/capacity/lead_time)",
        requested_aggregate="Wholesaler service areas: which wholesalers supply which "
        "pharmacies/hospitals, and shipment volume per lane",
    ),
    DataRequestItem(
        priority="2",
        topic="Software vendor identity",
        meeting_signal="~70% of BG alerts come from one software company",
        simulation_today="No vendor tag on locations; cannot reproduce the 70%",
        engine_parameter="new per-location software_vendor_id field",
        requested_aggregate="Alert counts grouped by originating software vendor "
        "(anonymized vendor codes)",
    ),
    DataRequestItem(
        priority="2",
        topic="Time-of-day / day-of-week",
        meeting_signal="Wholesaler scanning a valuable medicine at 23:00 is suspicious",
        simulation_today="tick == 1 day; no intra-day clock or weekday seasonality",
        engine_parameter="ticks_per_day, scan scheduling multipliers",
        requested_aggregate="Scan volume by hour-of-day and day-of-week (seasonality "
        "multipliers)",
    ),
    DataRequestItem(
        priority="2",
        topic="Medicine value and form",
        meeting_signal="Risk depends on medicine type and form (tablets, syringes)",
        simulation_today="Single synthetic product; no value or form attribute",
        engine_parameter="product value, product form, product_demand_weight",
        requested_aggregate="ATC-level volume shares, pack value bands, and dosage "
        "form per product",
    ),
    DataRequestItem(
        priority="2",
        topic="OBP -> wholesaler timing",
        meeting_signal="Benchmark: 95% sold within 3 months from OBP to wholesalers",
        simulation_today="Delivery timing is a synthetic MOVE proxy",
        engine_parameter="edge_lead_time_ticks, OBP supply capacity",
        requested_aggregate="Distribution of time from release to wholesaler receipt "
        "by organization and medicine type",
    ),
    DataRequestItem(
        priority="3",
        topic="Bad-actor / blacklist",
        meeting_signal="Certain bad actors -> trust-score debuff, a priori high trust",
        simulation_today="Trust score = geocode quality; no runtime blacklist",
        engine_parameter="location trust/quality score, blacklist flag",
        requested_aggregate="List (or risk tier) of sanctioned or previously "
        "implicated actors to seed a trust debuff",
    ),
)


def to_markdown_table(items: tuple[DataRequestItem, ...] = EMVO_DATA_REQUEST) -> str:
    header = (
        "| Priority | Topic | Meeting signal | Simulation today | Engine parameter | "
        "Requested aggregate |\n"
        "|---|---|---|---|---|---|"
    )
    lines = [header]
    for item in items:
        lines.append(
            f"| {item.priority} | {item.topic} | {item.meeting_signal} | "
            f"{item.simulation_today} | `{item.engine_parameter}` | "
            f"{item.requested_aggregate} |"
        )
    return "\n".join(lines)
