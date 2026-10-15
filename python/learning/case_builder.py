from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from canonical.models import CanonicalDataset
from compiler.enums import decode_org_type_u8
from compiler.types import EngineInput
from policy.models import Scenario

from .models import AlertCase


LOCAL_ROLES = {"LOCAL_ORG", "PHARMACY", "HOSPITAL"}
STANDARD_DECOMMISSION_REASON = "DISPENSED"


@dataclass(frozen=True)
class CaseBuilderConfig:
    safe_sample_rate: float = 0.10
    safe_sample_seed: int = 1729
    recent_window_ticks: int = 25
    recent_source_candidate_threshold: int = 25


@dataclass(frozen=True)
class LocationFacts:
    location_id: int
    ext_id: str
    org_id: int | None
    org_ext_id: str | None
    role: str | None
    market: str | None
    city: str | None
    postal_code: str | None
    latitude: float | None
    longitude: float | None
    trust_score: float


class EventHistory:
    def __init__(self, *, recent_window_ticks: int = 25):
        self.recent_window_ticks = int(recent_window_ticks)
        self._pack_events: Counter[int] = Counter()
        self._pack_alerts: Counter[int] = Counter()
        self._location_event_ticks: dict[int, deque[int]] = defaultdict(deque)

    def pack_event_count(self, pack_id: int | None) -> int:
        if pack_id is None:
            return 0
        return int(self._pack_events[int(pack_id)])

    def pack_alert_count(self, pack_id: int | None) -> int:
        if pack_id is None:
            return 0
        return int(self._pack_alerts[int(pack_id)])

    def recent_event_count(self, location_id: int | None, tick: int) -> int:
        if location_id is None:
            return 0
        low = int(tick) - self.recent_window_ticks
        values = self._location_event_ticks[int(location_id)]
        while values and values[0] < low:
            values.popleft()
        return len(values)

    def update(self, events: Sequence[Mapping[str, Any]]) -> None:
        for event in events:
            tick = int(event["tick"])
            pack_id = event.get("pack_id")
            if pack_id is not None:
                pid = int(pack_id)
                self._pack_events[pid] += 1
                if event.get("event_type") == "ALERT":
                    self._pack_alerts[pid] += 1
            locs = {
                loc
                for loc in (
                    event.get("from_location_id"),
                    event.get("to_location_id"),
                )
                if loc is not None
            }
            for loc in locs:
                values = self._location_event_ticks[int(loc)]
                values.append(tick)
                low = tick - self.recent_window_ticks
                while values and values[0] < low:
                    values.popleft()


class CaseBuilder:
    def __init__(
        self,
        inp: EngineInput,
        scenario: Scenario | None = None,
        canonical: CanonicalDataset | None = None,
        *,
        config: CaseBuilderConfig | None = None,
    ):
        self.inp = inp
        self.scenario = scenario
        self.canonical = canonical
        self.config = config or CaseBuilderConfig()
        self._facts = self._build_location_facts()

    def new_history(self) -> EventHistory:
        return EventHistory(recent_window_ticks=self.config.recent_window_ticks)

    def location_facts(self) -> dict[int, LocationFacts]:
        """Public view of per-location facts (role, geo, trust) keyed by location id."""
        return self._facts

    def build(
        self,
        events: Sequence[Mapping[str, Any]],
        history: EventHistory,
        *,
        stream_id: str = "stream",
    ) -> list[AlertCase]:
        out: list[AlertCase] = []
        for ordinal, event in enumerate(events):
            event_with_ordinal = dict(event)
            event_with_ordinal.setdefault("event_index", ordinal)
            if not self._is_candidate(event_with_ordinal, history, stream_id=stream_id):
                continue
            out.append(self._case_from_event(event_with_ordinal, history, stream_id=stream_id))
        return out

    def _is_candidate(
        self,
        event: Mapping[str, Any],
        history: EventHistory,
        *,
        stream_id: str,
    ) -> bool:
        event_type = str(event.get("event_type"))
        if event_type == "ALERT":
            return True
        if event_type == "DECOMMISSION":
            reason = _optional_str(event.get("decommission_reason"))
            if reason is not None and reason != STANDARD_DECOMMISSION_REASON:
                return True
        if event_type == "MOVE":
            src = self._facts.get(_optional_int(event.get("from_location_id")))
            dst = self._facts.get(_optional_int(event.get("to_location_id")))
            if src is not None and dst is not None:
                if src.market is not None and dst.market is not None and src.market != dst.market:
                    return True
                if _is_local_role(src.role) and _is_local_role(dst.role):
                    return True
            source_recent = history.recent_event_count(
                _optional_int(event.get("from_location_id")),
                int(event["tick"]),
            )
            if source_recent > self.config.recent_source_candidate_threshold:
                return True
        return self._safe_sampler_selects(event, stream_id=stream_id)

    def _case_from_event(
        self,
        event: Mapping[str, Any],
        history: EventHistory,
        *,
        stream_id: str,
    ) -> AlertCase:
        tick = int(event["tick"])
        pack_id = _optional_int(event.get("pack_id"))
        from_id = _optional_int(event.get("from_location_id"))
        to_id = _optional_int(event.get("to_location_id"))
        src = self._facts.get(from_id)
        dst = self._facts.get(to_id)
        distance = _distance_km(src, dst)
        same_postal = _same_optional(src.postal_code if src else None, dst.postal_code if dst else None)
        same_city = _same_optional(src.city if src else None, dst.city if dst else None)
        if src is not None and dst is not None and src.city is None and dst.city is None:
            same_city = same_postal
        same_org = (
            src is not None
            and dst is not None
            and src.org_id is not None
            and dst.org_id is not None
            and src.org_id == dst.org_id
        )
        from_market = src.market if src else None
        to_market = dst.market if dst else None
        event_index = int(event.get("event_index", 0))
        return AlertCase(
            case_id=_case_id(stream_id, event, event_index),
            tick=tick,
            pack_id=pack_id,
            event_type=str(event.get("event_type")),
            alert_code=_optional_str(event.get("alert_code")),
            decommission_reason=_optional_str(event.get("decommission_reason")),
            from_location_id=from_id,
            to_location_id=to_id,
            from_role=src.role if src else None,
            to_role=dst.role if dst else None,
            from_market=from_market,
            to_market=to_market,
            same_location=from_id is not None and to_id is not None and from_id == to_id,
            same_organization=same_org,
            same_city=same_city,
            same_postal_code=same_postal,
            distance_km=distance,
            market_mismatch=from_market is not None and to_market is not None and from_market != to_market,
            prior_pack_events=history.pack_event_count(pack_id),
            prior_pack_alerts=history.pack_alert_count(pack_id),
            source_recent_event_count=history.recent_event_count(from_id, tick),
            destination_recent_event_count=history.recent_event_count(to_id, tick),
            source_trust_score=src.trust_score if src else 1.0,
            destination_trust_score=dst.trust_score if dst else 1.0,
        )

    def _build_location_facts(self) -> dict[int, LocationFacts]:
        scenario_loc_by_ext = {}
        scenario_org_by_ext = {}
        if self.scenario is not None:
            scenario_loc_by_ext = {loc.ext_id: loc for loc in self.scenario.locations}
            scenario_org_by_ext = {org.ext_id: org for org in self.scenario.organizations}
        canonical_by_node = {}
        if self.canonical is not None:
            canonical_by_node = {node.node_id: node for node in self.canonical.nodes}

        facts: dict[int, LocationFacts] = {}
        org_type_by_location = np.asarray(self.inp.org_type)[np.asarray(self.inp.location_org_id)]
        for loc_id in range(self.inp.n_locations):
            ext_id = _location_ext_id(self.inp, loc_id)
            scenario_loc = scenario_loc_by_ext.get(ext_id)
            org_id = int(self.inp.location_org_id[loc_id])
            org_ext_id = None
            if org_id < len(self.inp.org_ext_id):
                org_ext_id = self.inp.org_ext_id[org_id]
            elif scenario_loc is not None:
                org_ext_id = scenario_loc.org_ext_id
            node = canonical_by_node.get(_node_id_from_location_ext_id(ext_id))
            if node is not None:
                role = node.role.value
                city = node.city or None
                postal_code = node.postal_code or None
                trust_score = float(node.quality_score)
                latitude = node.latitude
                longitude = node.longitude
            else:
                role = decode_org_type_u8(int(org_type_by_location[loc_id])).value
                city = None
                postal_code = scenario_loc.postal_code if scenario_loc is not None else None
                trust_score = 1.0
                latitude = None
                longitude = None
            market_id = int(self.inp.location_market_id[loc_id])
            market = self.inp.market_code[market_id] if market_id < len(self.inp.market_code) else None
            if scenario_loc is not None:
                market = scenario_loc.market_code or market
                if org_ext_id is None:
                    org_ext_id = scenario_loc.org_ext_id
                if scenario_org_by_ext.get(scenario_loc.org_ext_id) is not None and node is None:
                    role = scenario_org_by_ext[scenario_loc.org_ext_id].org_type.value
            facts[loc_id] = LocationFacts(
                location_id=loc_id,
                ext_id=ext_id,
                org_id=org_id,
                org_ext_id=org_ext_id,
                role=role,
                market=market,
                city=city,
                postal_code=postal_code,
                latitude=latitude,
                longitude=longitude,
                trust_score=max(0.0, min(1.0, trust_score)),
            )
        return facts

    def _safe_sampler_selects(self, event: Mapping[str, Any], *, stream_id: str) -> bool:
        rate = self.config.safe_sample_rate
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return _hash_unit(str(self.config.safe_sample_seed), stream_id, _event_key(event)) < rate


def build_cases_from_records(
    records: Sequence[Mapping[str, Any]],
    builder: CaseBuilder,
    *,
    stream_id: str = "stream",
) -> list[AlertCase]:
    history = builder.new_history()
    out: list[AlertCase] = []
    by_tick: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_tick[int(row["tick"])].append(row)
    for tick in sorted(by_tick):
        tick_events = [dict(row, event_index=i) for i, row in enumerate(by_tick[tick])]
        out.extend(builder.build(tick_events, history, stream_id=stream_id))
        history.update(tick_events)
    return out


def _location_ext_id(inp: EngineInput, loc_id: int) -> str:
    if inp.location_ext_id and loc_id < len(inp.location_ext_id):
        return inp.location_ext_id[loc_id]
    return f"loc#{loc_id}"


def _node_id_from_location_ext_id(ext_id: str) -> str:
    return ext_id[len("loc_") :] if ext_id.startswith("loc_") else ext_id


def _case_id(stream_id: str, event: Mapping[str, Any], event_index: int) -> str:
    pack = event.get("pack_id")
    src = event.get("from_location_id")
    dst = event.get("to_location_id")
    return (
        f"{stream_id}:t{int(event['tick'])}:e{event_index}:"
        f"{event.get('event_type')}:p{pack if pack is not None else 'none'}:"
        f"{src if src is not None else 'none'}:{dst if dst is not None else 'none'}"
    )


def _event_key(event: Mapping[str, Any]) -> str:
    return "|".join(
        str(event.get(k))
        for k in (
            "tick",
            "event_index",
            "pack_id",
            "event_type",
            "from_location_id",
            "to_location_id",
            "alert_code",
            "decommission_reason",
        )
    )


def _hash_unit(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _same_optional(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a == b


def _is_local_role(role: str | None) -> bool:
    return role in LOCAL_ROLES


def _distance_km(a: LocationFacts | None, b: LocationFacts | None) -> float | None:
    if a is None or b is None:
        return None
    if a.location_id == b.location_id:
        return 0.0
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    lat1 = math.radians(float(a.latitude))
    lat2 = math.radians(float(b.latitude))
    dlat = lat2 - lat1
    dlon = math.radians(float(b.longitude) - float(a.longitude))
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(h))
