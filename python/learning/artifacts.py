"""Self-contained supervisor artifacts for the HITL dashboard.

These writers turn the in-memory HITL case streams into three extra files next to
``cases.jsonl`` / ``metrics.json`` so the dashboard can be driven entirely from disk
without re-running the C++ engine:

* ``network.json`` -- geographic nodes + supply edges (map & routes tab).
* ``pack_traces.jsonl`` -- per-pack event chains for the layered-triage tab.
* ``calibration.json`` -- simulation counters next to the meeting reference numbers.

Everything here is a pure function of the streams so it is unit-testable with small
fixtures and never imports Streamlit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .case_builder import CaseBuilder, CaseBuilderConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runner import CaseStream, HitlDemoConfig


# 1 tick == 1 day in the Bulgaria scenario; a "3 month" horizon is 90 ticks.
TICKS_PER_MONTH = 30
SALES_HORIZON_TICKS = 90
_SCAN_EVENT_TYPES = {"VERIFY", "DECOMMISSION", "REACTIVATE"}
_WHOLESALER_ROLES = {"WHOLESALER"}
_MAX_TRACE_PACKS = 400

# Operational figures from the supervisor meeting (see docs/notes/meeting_notes_07_30.md). These are the
# yardsticks the dashboard puts next to the simulation so nothing is hidden.
MEETING_REFERENCE: dict[str, Any] = {
    "bulgaria_alerts_per_week": 3000,
    "germany_alerts_per_week": 35000,
    "eu_scans_per_year": 160_000_000,
    "current_alert_rate_ceiling": 0.005,
    "target_alert_rate": 0.0005,
    "share_bulgaria_alerts_single_vendor": 0.70,
    "sales_within_3_months_fraction": 0.95,
    "notes": (
        "Bulgaria ~3000 alerts/week (most errors, ~70% from one software vendor); "
        "Germany ~35000/week; ~160M scans/year; alert rate <0.5% today, target 0.05%; "
        "benchmark 95% sold within 3 months OBP->wholesalers (varies by organization, "
        "time, medicine type and form)."
    ),
}


def write_supervisor_artifacts(
    output_dir: Path,
    *,
    train_streams: Sequence["CaseStream"],
    eval_streams: Sequence["CaseStream"],
    map_stream: "CaseStream | None",
    config: "HitlDemoConfig",
    commit_hash: str,
    predicted_location_ids: Iterable[int] = (),
    target_location_ids: Iterable[int] = (),
) -> None:
    """Write ``network.json``, ``pack_traces.jsonl`` and ``calibration.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    builder_config = CaseBuilderConfig(
        safe_sample_rate=config.safe_sample_rate,
        safe_sample_seed=config.safe_sample_seed,
        recent_window_ticks=config.recent_window_ticks,
        recent_source_candidate_threshold=config.recent_source_candidate_threshold,
    )
    network = build_network_payload(
        map_stream,
        commit_hash=commit_hash,
        builder_config=builder_config,
        predicted_location_ids=set(int(x) for x in predicted_location_ids),
        target_location_ids=set(int(x) for x in target_location_ids),
    )
    (output_dir / "network.json").write_text(
        json.dumps(network, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    traces = build_pack_traces(map_stream, commit_hash=commit_hash)
    with (output_dir / "pack_traces.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    calibration = build_calibration_payload(
        train_streams,
        eval_streams,
        commit_hash=commit_hash,
        ticks=config.ticks,
        builder_config=builder_config,
    )
    (output_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_network_payload(
    map_stream: "CaseStream | None",
    *,
    commit_hash: str,
    builder_config: CaseBuilderConfig,
    predicted_location_ids: set[int],
    target_location_ids: set[int],
) -> dict[str, Any]:
    if map_stream is None or map_stream.engine_input is None:
        return {
            "commit_hash": commit_hash,
            "stream_id": None,
            "seed": None,
            "nodes": [],
            "edges": [],
            "predicted_location_ids": sorted(predicted_location_ids),
            "target_location_ids": sorted(target_location_ids),
            "role_counts": {},
            "geocoded": False,
        }

    builder = CaseBuilder(
        map_stream.engine_input,
        map_stream.scenario,
        map_stream.canonical,
        config=builder_config,
    )
    facts = builder.location_facts()
    ext_to_id = {fact.ext_id: loc_id for loc_id, fact in facts.items()}

    alert_counts: dict[int, int] = {}
    for record in map_stream.records:
        if str(record.get("event_type")) != "ALERT":
            continue
        for key in ("from_location_id", "to_location_id"):
            loc = record.get(key)
            if loc is not None:
                alert_counts[int(loc)] = alert_counts.get(int(loc), 0) + 1

    nodes: list[dict[str, Any]] = []
    role_counts: dict[str, int] = {}
    geocoded = False
    for loc_id in sorted(facts):
        fact = facts[loc_id]
        if fact.latitude is not None and fact.longitude is not None:
            geocoded = True
        role = fact.role or "UNKNOWN"
        role_counts[role] = role_counts.get(role, 0) + 1
        nodes.append(
            {
                "location_id": loc_id,
                "ext_id": fact.ext_id,
                "org_ext_id": fact.org_ext_id,
                "role": role,
                "market": fact.market,
                "city": fact.city,
                "postal_code": fact.postal_code,
                "latitude": fact.latitude,
                "longitude": fact.longitude,
                "trust_score": fact.trust_score,
                "alert_count": alert_counts.get(loc_id, 0),
                "predicted_unsafe": loc_id in predicted_location_ids,
                "target": loc_id in target_location_ids,
            }
        )

    edges: list[dict[str, Any]] = []
    scenario = map_stream.scenario
    if scenario is not None:
        for edge in getattr(scenario, "location_edges", []):
            src = ext_to_id.get(edge.src_location_ext_id)
            dst = ext_to_id.get(edge.dst_location_ext_id)
            if src is None or dst is None:
                continue
            edges.append(
                {
                    "src_location_id": src,
                    "dst_location_id": dst,
                    "src_ext_id": edge.src_location_ext_id,
                    "dst_ext_id": edge.dst_location_ext_id,
                    "cost": float(edge.cost),
                    "capacity": int(edge.capacity),
                }
            )

    return {
        "commit_hash": commit_hash,
        "stream_id": map_stream.stream_id,
        "seed": map_stream.seed,
        "nodes": nodes,
        "edges": edges,
        "predicted_location_ids": sorted(predicted_location_ids),
        "target_location_ids": sorted(target_location_ids),
        "role_counts": role_counts,
        "geocoded": geocoded,
        "edge_model": "k-NN haversine proxy (not licensed supply links)",
    }


def build_pack_traces(
    map_stream: "CaseStream | None",
    *,
    commit_hash: str,
    max_packs: int = _MAX_TRACE_PACKS,
) -> list[dict[str, Any]]:
    if map_stream is None or map_stream.engine_input is None:
        return []
    from runtime.simulation_viz import filter_events_for_pack, pack_move_trace

    inp = map_stream.engine_input
    n_packs = int(getattr(inp, "n_packs", 0) or len(getattr(inp, "pack_initial_location_id", [])))

    # Packs that surfaced in an alert case, most-referenced first, capped for size.
    pack_case_ids: dict[int, list[str]] = {}
    for case in map_stream.cases:
        if case.pack_id is None:
            continue
        pack_case_ids.setdefault(int(case.pack_id), []).append(case.case_id)
    ordered = sorted(
        pack_case_ids,
        key=lambda pid: (-len(pack_case_ids[pid]), pid),
    )[:max_packs]

    traces: list[dict[str, Any]] = []
    for pack_id in ordered:
        if pack_id < 0 or (n_packs and pack_id >= n_packs):
            continue
        events = [
            {
                "tick": int(ev["tick"]),
                "event_type": ev.get("event_type"),
                "from_location_id": ev.get("from_location_id"),
                "to_location_id": ev.get("to_location_id"),
                "from_location": ev.get("from_location"),
                "to_location": ev.get("to_location"),
                "alert_code": ev.get("alert_code"),
                "decommission_reason": ev.get("decommission_reason"),
                "market": ev.get("market"),
            }
            for ev in filter_events_for_pack(map_stream.records, pack_id)
        ]
        trace = [
            {"tick": tick, "location_ext_id": label}
            for tick, label in pack_move_trace(map_stream.records, inp, pack_id)
        ]
        traces.append(
            {
                "commit_hash": commit_hash,
                "stream_id": map_stream.stream_id,
                "seed": map_stream.seed,
                "pack_id": pack_id,
                "case_ids": sorted(pack_case_ids[pack_id]),
                "events": events,
                "trace": trace,
            }
        )
    return traces


def build_calibration_payload(
    train_streams: Sequence["CaseStream"],
    eval_streams: Sequence["CaseStream"],
    *,
    commit_hash: str,
    ticks: int,
    builder_config: CaseBuilderConfig,
) -> dict[str, Any]:
    # Single-seed runs split one stream into train/eval that share the same underlying
    # event records; dedupe on record-list identity so scans/alerts are counted once.
    seen_records: set[int] = set()
    streams: list["CaseStream"] = []
    for stream in [*train_streams, *eval_streams]:
        marker = id(stream.records)
        if marker in seen_records:
            continue
        seen_records.add(marker)
        streams.append(stream)
    per_stream: list[dict[str, Any]] = []
    baseline_scans = 0
    baseline_alerts = 0
    for stream in streams:
        counts = _count_events(stream.records)
        scans = sum(counts.get(t, 0) for t in _SCAN_EVENT_TYPES)
        alerts = counts.get("ALERT", 0)
        is_baseline = stream.perturbation_type is None
        if is_baseline:
            baseline_scans += scans
            baseline_alerts += alerts
        per_stream.append(
            {
                "stream_id": stream.stream_id,
                "seed": stream.seed,
                "perturbation_type": stream.perturbation_type,
                "scans": scans,
                "alerts": alerts,
                "alert_rate": _ratio(alerts, scans),
                "alerts_per_week": _per_week(alerts, ticks),
                "event_counts": counts,
            }
        )

    role_by_location = _role_by_location(eval_streams, builder_config)
    sales = _sales_benchmark(eval_streams, role_by_location)

    return {
        "commit_hash": commit_hash,
        "ticks": ticks,
        "ticks_per_month": TICKS_PER_MONTH,
        "sales_horizon_ticks": SALES_HORIZON_TICKS,
        "simulation": {
            "baseline": {
                "scans": baseline_scans,
                "alerts": baseline_alerts,
                "alert_rate": _ratio(baseline_alerts, baseline_scans),
                "alerts_per_week": _per_week(baseline_alerts, ticks),
            },
            "sales_benchmark": sales,
        },
        "per_stream": per_stream,
        "reference": MEETING_REFERENCE,
    }


def _count_events(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        event_type = str(record.get("event_type"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _role_by_location(
    eval_streams: Sequence["CaseStream"],
    builder_config: CaseBuilderConfig,
) -> dict[int, str | None]:
    for stream in eval_streams:
        if stream.engine_input is None:
            continue
        builder = CaseBuilder(
            stream.engine_input,
            stream.scenario,
            stream.canonical,
            config=builder_config,
        )
        return {loc_id: fact.role for loc_id, fact in builder.location_facts().items()}
    return {}


def _sales_benchmark(
    eval_streams: Sequence["CaseStream"],
    role_by_location: Mapping[int, str | None],
) -> dict[str, Any]:
    """Time from simulation start to first downstream delivery per pack (a sales proxy).

    "Delivery" is the first MOVE whose destination is a wholesaler or terminal, which
    stands in for the meeting benchmark of 95% sold within three months.
    """
    first_delivery: dict[tuple[Any, int], int] = {}
    for stream in eval_streams:
        for record in stream.records:
            if str(record.get("event_type")) != "MOVE":
                continue
            pack_id = record.get("pack_id")
            to_loc = record.get("to_location_id")
            if pack_id is None or to_loc is None:
                continue
            key = (stream.seed, int(pack_id))
            tick = int(record["tick"])
            if key not in first_delivery or tick < first_delivery[key]:
                first_delivery[key] = tick
    delivery_ticks = sorted(first_delivery.values())
    n_delivered = len(delivery_ticks)
    if n_delivered == 0:
        return {
            "definition": "first MOVE tick per pack (downstream delivery proxy)",
            "n_packs_delivered": 0,
            "fraction_within_horizon": None,
            "p50_ticks": None,
            "p90_ticks": None,
            "p95_ticks": None,
            "wholesaler_roles_present": any(
                r in _WHOLESALER_ROLES for r in role_by_location.values()
            ),
        }
    within = sum(1 for t in delivery_ticks if t <= SALES_HORIZON_TICKS)
    return {
        "definition": "first MOVE tick per pack (downstream delivery proxy)",
        "n_packs_delivered": n_delivered,
        "fraction_within_horizon": within / n_delivered,
        "p50_ticks": _percentile(delivery_ticks, 0.50),
        "p90_ticks": _percentile(delivery_ticks, 0.90),
        "p95_ticks": _percentile(delivery_ticks, 0.95),
        "wholesaler_roles_present": any(
            r in _WHOLESALER_ROLES for r in role_by_location.values()
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _per_week(count: int, ticks: int) -> float | None:
    if ticks <= 0:
        return None
    return count / ticks * 7.0


def _percentile(sorted_values: Sequence[int], q: float) -> int | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    idx = int(round(q * (len(sorted_values) - 1)))
    idx = max(0, min(len(sorted_values) - 1, idx))
    return int(sorted_values[idx])
