"""
Inspect, export, and (optionally) plot native simulation state and event logs.

Use with a compiled :class:`~compiler.types.EngineInput` and a native
:class:`Simulator` from :mod:`runtime.native_bridge`.

**During-run debugging:** :func:`run_ticks_with_hook` advances one tick at a time and
calls your callback so you can print or log deltas.

**Plots:** install optional deps ``pip install pharmasim[viz]`` (matplotlib), then use
:func:`plot_event_timeline` / :func:`plot_physical_locations`.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, TextIO

from compiler.enums import (
    U8_TO_ALERT_CODE,
    U8_TO_DECOMMISSION_REASON,
    U8_TO_PACK_STATE,
)
from compiler.types import EngineInput

# Match C++ k_event_no_location / k_event_no_market / UINT32_MAX
NO_LOCATION: int = 2**32 - 1
NO_MARKET: int = 2**32 - 1

EVENT_TYPE_NAMES: dict[int, str] = {
    0: "VERIFY",
    1: "DECOMMISSION",
    2: "REACTIVATE",
    3: "MOVE",
    4: "REGISTRY_SYNC",
    5: "ALERT",
}

# event_type_u8 values whose event_detail column carries meaning.
_DECOMMISSION_U8 = 1
_ALERT_U8 = 5


class NativeSimulator(Protocol):
    def current_tick(self) -> int: ...
    def event_count(self) -> int: ...
    def event_log_ticks(self) -> Sequence[int]: ...
    def event_log_pack_ids(self) -> Sequence[int]: ...
    def event_log_types(self) -> Sequence[int]: ...
    def event_log_from_locations(self) -> Sequence[int]: ...
    def event_log_to_locations(self) -> Sequence[int]: ...
    def event_log_details(self) -> Sequence[int]: ...
    def event_log_market_ids(self) -> Sequence[int]: ...
    def physical_pack_states(self) -> Sequence[int]: ...
    def physical_pack_location_ids(self) -> Sequence[int]: ...
    def physical_pack_market_ids(self) -> Sequence[int]: ...
    def registry_matches_physical(self) -> bool: ...
    def run_ticks(self, n: int) -> None: ...


def _loc_label(inp: EngineInput, loc_id: int) -> str:
    if loc_id == NO_LOCATION or loc_id < 0:
        return "—"
    if loc_id >= len(inp.location_ext_id):
        return f"loc#{loc_id}"
    return inp.location_ext_id[loc_id]


def _pack_label(inp: EngineInput, pack_id: int) -> str:
    if pack_id < 0:
        return "—"
    if pack_id < len(inp.pack_ext_id) and inp.pack_ext_id[pack_id]:
        return inp.pack_ext_id[pack_id]
    return f"pack#{pack_id}"


def _market_label(inp: EngineInput, market_id: int) -> str:
    if market_id < 0 or market_id >= len(inp.market_code):
        return f"market#{market_id}"
    return inp.market_code[market_id]


def _event_name(code: int) -> str:
    return EVENT_TYPE_NAMES.get(code, f"TYPE_{code}")


def _detail_label(event_type_u8: int, detail_u8: int) -> str | None:
    """Decode the overloaded ``event_detail`` column by event type."""
    if event_type_u8 == _DECOMMISSION_U8:
        reason = U8_TO_DECOMMISSION_REASON.get(detail_u8)
        return str(reason) if reason is not None else None
    if event_type_u8 == _ALERT_U8:
        code = U8_TO_ALERT_CODE.get(detail_u8)
        return str(code) if code is not None else None
    return None


def events_as_records(sim: NativeSimulator, inp: EngineInput) -> list[dict[str, Any]]:
    """One dict per event row (aligned with C++ :class:`EventLog`).

    EMVS columns: ``alert_code`` is set on ALERT rows, ``decommission_reason`` on DECOMMISSION
    rows, ``market_id``/``market`` give the national NMVS the transaction hit.
    """
    ticks = list(sim.event_log_ticks())
    pack_ids = list(sim.event_log_pack_ids())
    types = list(sim.event_log_types())
    from_locs = list(sim.event_log_from_locations())
    to_locs = list(sim.event_log_to_locations())
    details = list(sim.event_log_details())
    markets = list(sim.event_log_market_ids())
    n = len(ticks)
    if not (
        len(pack_ids) == len(types) == len(from_locs) == len(to_locs)
        == len(details) == len(markets) == n
    ):
        raise RuntimeError("event log columns have mismatched lengths")
    rows: list[dict[str, Any]] = []
    for i in range(n):
        pid = int(pack_ids[i])
        fl = int(from_locs[i])
        tl = int(to_locs[i])
        et_u8 = int(types[i])
        detail_u8 = int(details[i])
        mkt = int(markets[i])
        detail = _detail_label(et_u8, detail_u8)
        rows.append(
            {
                "tick": int(ticks[i]),
                "pack_id": pid,
                "pack_ext_id": _pack_label(inp, pid),
                "event_type": _event_name(et_u8),
                "event_type_u8": et_u8,
                "from_location_id": fl if fl != NO_LOCATION else None,
                "to_location_id": tl if tl != NO_LOCATION else None,
                "from_location": _loc_label(inp, fl),
                "to_location": _loc_label(inp, tl),
                "market_id": mkt if mkt != NO_MARKET else None,
                "market": _market_label(inp, mkt) if mkt != NO_MARKET else None,
                "alert_code": detail if et_u8 == _ALERT_U8 else None,
                "decommission_reason": detail if et_u8 == _DECOMMISSION_U8 else None,
            }
        )
    return rows


def physical_snapshot_rows(sim: NativeSimulator, inp: EngineInput) -> list[dict[str, Any]]:
    """Current physical SoA state per pack (after last completed tick)."""
    states = list(sim.physical_pack_states())
    locs = list(sim.physical_pack_location_ids())
    mkts = list(sim.physical_pack_market_ids())
    n = len(states)
    if len(locs) != n or len(mkts) != n:
        raise RuntimeError("physical pack columns have mismatched lengths")
    rows: list[dict[str, Any]] = []
    for i in range(n):
        su8 = int(states[i])
        st = U8_TO_PACK_STATE.get(su8, None)
        rows.append(
            {
                "pack_id": i,
                "pack_ext_id": _pack_label(inp, i),
                "state_u8": su8,
                "state": st.value if st is not None else f"UNKNOWN_{su8}",
                "location_id": int(locs[i]),
                "location_ext_id": _loc_label(inp, int(locs[i])),
                "market_id": int(mkts[i]),
                "market_code": _market_label(inp, int(mkts[i])),
            }
        )
    return rows


def filter_events_for_pack(records: Iterable[Mapping[str, Any]], pack_id: int) -> list[dict[str, Any]]:
    return [dict(r) for r in records if int(r["pack_id"]) == pack_id]


def format_events_text(
    records: Sequence[Mapping[str, Any]],
    *,
    max_rows: int | None = None,
) -> str:
    """Fixed-width style table for terminals."""
    cols = ("tick", "pack_ext_id", "event_type", "from_location", "to_location")
    lines = ["\t".join(cols)]
    for i, r in enumerate(records):
        if max_rows is not None and i >= max_rows:
            lines.append(f"... ({len(records) - max_rows} more rows)")
            break
        lines.append(
            "\t".join(
                str(r[c]) for c in cols
            )
        )
    return "\n".join(lines)


def format_snapshot_text(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = ("pack_ext_id", "state", "location_ext_id", "market_code")
    out = ["\t".join(cols)]
    for r in rows:
        out.append("\t".join(str(r[c]) for c in cols))
    return "\n".join(out)


def write_events_csv(records: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    path = Path(path)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


def write_events_jsonl(records: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_snapshot_json(snapshot: Sequence[Mapping[str, Any]], path: Path | str) -> None:
    Path(path).write_text(
        json.dumps(list(snapshot), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class TickDebugInfo:
    tick_index: int
    simulator_tick: int
    new_events: list[dict[str, Any]]
    snapshot: list[dict[str, Any]]
    registry_ok: bool


def run_ticks_with_hook(
    sim: NativeSimulator,
    inp: EngineInput,
    n: int,
    on_tick: Callable[[TickDebugInfo], None],
) -> None:
    """
    Advance ``n`` ticks, calling ``on_tick`` after each native tick (uses ``run_ticks(1)``).

    ``TickDebugInfo.new_events`` contains only rows appended since the previous tick.
    """
    for i in range(n):
        prev_n = sim.event_count()
        sim.run_ticks(1)
        all_rows = events_as_records(sim, inp)
        new_events = all_rows[prev_n:] if prev_n <= len(all_rows) else []
        info = TickDebugInfo(
            tick_index=i,
            simulator_tick=int(sim.current_tick()),
            new_events=new_events,
            snapshot=physical_snapshot_rows(sim, inp),
            registry_ok=bool(sim.registry_matches_physical()),
        )
        on_tick(info)


def print_tick_debug(
    info: TickDebugInfo,
    *,
    stream: TextIO = sys.stdout,
    max_event_rows: int = 50,
) -> None:
    """Default hook: print tick header, new events (truncated), and snapshot."""
    print(f"--- tick {info.tick_index} (simulator current_tick={info.simulator_tick}) ---", file=stream)
    print(f"registry_matches_physical={info.registry_ok}", file=stream)
    if info.new_events:
        print("new events:", file=stream)
        print(format_events_text(info.new_events, max_rows=max_event_rows), file=stream)
    else:
        print("new events: (none)", file=stream)
    print("physical snapshot:", file=stream)
    print(format_snapshot_text(info.snapshot), file=stream)
    print(file=stream)


def export_run_report(
    sim: NativeSimulator,
    inp: EngineInput,
    out_dir: Path | str,
    *,
    prefix: str = "run",
) -> None:
    """Write ``events.csv``, ``events.jsonl``, and ``snapshot_final.json`` under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = events_as_records(sim, inp)
    write_events_csv(ev, out / f"{prefix}_events.csv")
    write_events_jsonl(ev, out / f"{prefix}_events.jsonl")
    write_snapshot_json(physical_snapshot_rows(sim, inp), out / f"{prefix}_snapshot_final.json")


def plot_event_timeline(
    records: Sequence[Mapping[str, Any]],
    *,
    ax: Any = None,
    title: str = "Simulation events",
):
    """
    Scatter: x=tick, y=pack_id, one series per event_type (legend). Requires matplotlib.

    ``pip install pharmasim[viz]`` or ``pip install matplotlib``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_event_timeline requires matplotlib. Install with: pip install matplotlib"
        ) from e

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if not records:
        ax.set_title(title + " (no events)")
        fig.tight_layout()
        return fig, ax

    types = sorted({str(r["event_type"]) for r in records})
    for et in types:
        sub = [r for r in records if str(r["event_type"]) == et]
        ax.scatter(
            [int(r["tick"]) for r in sub],
            [int(r["pack_id"]) for r in sub],
            label=et,
            alpha=0.85,
        )
    ax.set_xlabel("tick")
    ax.set_ylabel("pack_id")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_physical_locations(
    sim: NativeSimulator,
    inp: EngineInput,
    *,
    ax: Any = None,
    title: str = "Packs by location (current)",
):
    """Bar chart: count of packs per location_ext_id. Requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_physical_locations requires matplotlib. Install with: pip install matplotlib"
        ) from e

    rows = physical_snapshot_rows(sim, inp)
    c = Counter(str(r["location_ext_id"]) for r in rows)
    labels = sorted(c.keys())
    vals = [c[k] for k in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.figure

    ax.bar(labels, vals)
    ax.set_ylabel("packs")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    fig.tight_layout()
    return fig, ax


def pack_move_trace(
    records: Sequence[Mapping[str, Any]],
    inp: EngineInput,
    pack_id: int,
) -> list[tuple[int, str]]:
    """
    Reconstruct (tick, location_ext_id) after each MOVE for one pack, including initial location.
    """
    initial = int(inp.pack_initial_location_id[pack_id])
    trace: list[tuple[int, str]] = [(-1, _loc_label(inp, initial))]
    loc = initial
    for r in records:
        if int(r["pack_id"]) != pack_id:
            continue
        if r["event_type"] != "MOVE":
            continue
        tl = r["to_location_id"]
        if tl is None:
            continue
        loc = int(tl)
        trace.append((int(r["tick"]), _loc_label(inp, loc)))
    return trace


def format_pack_history_text(
    records: Sequence[Mapping[str, Any]],
    inp: EngineInput,
    pack_id: int,
) -> str:
    """Human-readable MOVE trace + all events for one pack."""
    lines = [
        f"pack {_pack_label(inp, pack_id)} (id={pack_id})",
        "location trace (initial + after each MOVE):",
    ]
    for t, lab in pack_move_trace(records, inp, pack_id):
        lines.append(f"  tick {t}: {lab}")
    lines.append("all events:")
    for r in filter_events_for_pack(records, pack_id):
        lines.append(
            f"  tick {r['tick']} {r['event_type']} {r['from_location']} -> {r['to_location']}"
        )
    return "\n".join(lines)


def plot_location_graph(
    canonical: Any,
    scenario: Any,
    *,
    ax: Any = None,
    title: str = "Distribution graph",
    highlight_location_ext_ids: Iterable[str] | None = None,
):
    """Geographic plot of the distribution graph: nodes by org type + supply edges.

    ``canonical`` is a ``CanonicalDataset`` (nodes carry lat/lon); ``scenario`` provides the
    org type per location and the compiled edges. Optionally highlights flagged locations
    (e.g. detected fraud sites). Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_location_graph requires matplotlib. Install with: pip install matplotlib"
        ) from e

    coord_by_node = {
        n.node_id: (float(n.longitude), float(n.latitude))
        for n in canonical.nodes
        if n.latitude is not None and n.longitude is not None
    }

    def loc_coord(loc_ext_id: str) -> tuple[float, float] | None:
        node_id = loc_ext_id[len("loc_") :] if loc_ext_id.startswith("loc_") else loc_ext_id
        return coord_by_node.get(node_id)

    org_type_by_org = {o.ext_id: str(o.org_type) for o in scenario.organizations}
    type_by_loc = {
        loc.ext_id: org_type_by_org.get(loc.org_ext_id, "?") for loc in scenario.locations
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Edges first (under the nodes).
    for e in scenario.location_edges:
        a = loc_coord(e.src_location_ext_id)
        b = loc_coord(e.dst_location_ext_id)
        if a is None or b is None:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=0.3, alpha=0.35, zorder=1)

    groups = {"OBP": ([], [], 60.0, "*"), "WHOLESALER": ([], [], 18.0, "s"), "LOCAL_ORG": ([], [], 8.0, "o")}
    for loc in scenario.locations:
        c = loc_coord(loc.ext_id)
        if c is None:
            continue
        g = type_by_loc.get(loc.ext_id, "LOCAL_ORG")
        if g not in groups:
            g = "LOCAL_ORG"
        groups[g][0].append(c[0])
        groups[g][1].append(c[1])

    for label, (xs, ys, size, marker) in groups.items():
        if xs:
            ax.scatter(xs, ys, s=size, marker=marker, label=f"{label} ({len(xs)})", zorder=2, alpha=0.8)

    if highlight_location_ext_ids:
        hx, hy = [], []
        for loc_ext_id in highlight_location_ext_ids:
            c = loc_coord(loc_ext_id)
            if c is not None:
                hx.append(c[0])
                hy.append(c[1])
        if hx:
            ax.scatter(
                hx, hy, s=160, facecolors="none", edgecolors="red", linewidths=1.8,
                label="flagged", zorder=3,
            )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig, ax


def plot_flow_comparison(
    baseline_records: Sequence[Mapping[str, Any]],
    perturbed_records: Sequence[Mapping[str, Any]],
    *,
    ax: Any = None,
    title: str = "Baseline vs perturbed event volume",
):
    """Grouped bar chart comparing event-type counts between baseline and perturbed runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_flow_comparison requires matplotlib. Install with: pip install matplotlib"
        ) from e

    base_counts = Counter(str(r["event_type"]) for r in baseline_records)
    pert_counts = Counter(str(r["event_type"]) for r in perturbed_records)
    types = sorted(set(base_counts) | set(pert_counts))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    x = range(len(types))
    width = 0.4
    ax.bar([i - width / 2 for i in x], [base_counts.get(t, 0) for t in types], width, label="baseline")
    ax.bar([i + width / 2 for i in x], [pert_counts.get(t, 0) for t in types], width, label="perturbed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("event count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def dump_debug_report_to_string(sim: NativeSimulator, inp: EngineInput) -> str:
    """Full text report suitable for logging at end of run."""
    buf = StringIO()
    print(f"current_tick={sim.current_tick()}", file=buf)
    print(f"event_count={sim.event_count()}", file=buf)
    print(f"registry_matches_physical={sim.registry_matches_physical()}", file=buf)
    print(file=buf)
    print("=== events ===", file=buf)
    print(format_events_text(events_as_records(sim, inp)), file=buf)
    print(file=buf)
    print("=== physical snapshot ===", file=buf)
    print(format_snapshot_text(physical_snapshot_rows(sim, inp)), file=buf)
    return buf.getvalue()
