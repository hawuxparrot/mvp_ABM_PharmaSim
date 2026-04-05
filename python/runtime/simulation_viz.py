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

from compiler.enums import U8_TO_PACK_STATE
from compiler.types import EngineInput

# Match C++ k_event_no_location / UINT32_MAX
NO_LOCATION: int = 2**32 - 1

EVENT_TYPE_NAMES: dict[int, str] = {
    0: "VERIFY",
    1: "DECOMMISSION",
    2: "REACTIVATE",
    3: "MOVE",
    4: "REGISTRY_SYNC",
}


class NativeSimulator(Protocol):
    def current_tick(self) -> int: ...
    def event_count(self) -> int: ...
    def event_log_ticks(self) -> Sequence[int]: ...
    def event_log_pack_ids(self) -> Sequence[int]: ...
    def event_log_types(self) -> Sequence[int]: ...
    def event_log_from_locations(self) -> Sequence[int]: ...
    def event_log_to_locations(self) -> Sequence[int]: ...
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


def events_as_records(sim: NativeSimulator, inp: EngineInput) -> list[dict[str, Any]]:
    """One dict per event row (aligned with C++ :class:`EventLog`)."""
    ticks = list(sim.event_log_ticks())
    pack_ids = list(sim.event_log_pack_ids())
    types = list(sim.event_log_types())
    from_locs = list(sim.event_log_from_locations())
    to_locs = list(sim.event_log_to_locations())
    n = len(ticks)
    if not (len(pack_ids) == len(types) == len(from_locs) == len(to_locs) == n):
        raise RuntimeError("event log columns have mismatched lengths")
    rows: list[dict[str, Any]] = []
    for i in range(n):
        pid = int(pack_ids[i])
        fl = int(from_locs[i])
        tl = int(to_locs[i])
        rows.append(
            {
                "tick": int(ticks[i]),
                "pack_id": pid,
                "pack_ext_id": _pack_label(inp, pid),
                "event_type": _event_name(int(types[i])),
                "event_type_u8": int(types[i]),
                "from_location_id": fl if fl != NO_LOCATION else None,
                "to_location_id": tl if tl != NO_LOCATION else None,
                "from_location": _loc_label(inp, fl),
                "to_location": _loc_label(inp, tl),
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
