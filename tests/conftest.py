"""Shared pytest hooks for the test suite."""

from __future__ import annotations

from typing import Any


def _format_seconds(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds:,.2f}s ({seconds / 3600:.2f} h)"
    if seconds >= 60:
        return f"{seconds:,.2f}s ({seconds / 60:.1f} min)"
    if seconds >= 1:
        return f"{seconds:,.3f}s"
    return f"{seconds * 1000:,.1f} ms"


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    seen: set[str] = set()
    breakdown_reports: list[tuple[dict[str, float], dict[str, Any]]] = []
    for reports in terminalreporter.stats.values():
        for report in reports:
            nodeid = getattr(report, "nodeid", "")
            if nodeid in seen:
                continue
            props = dict(getattr(report, "user_properties", []))
            breakdown = props.get("bench_breakdown")
            meta = props.get("bench_meta")
            if breakdown is not None and meta is not None:
                seen.add(nodeid)
                breakdown_reports.append((breakdown, meta))

    if not breakdown_reports:
        return

    terminalreporter.write_sep("=", "Benchmark phase breakdown")
    for breakdown, meta in breakdown_reports:
        terminalreporter.write_line(f"Profile: {meta.get('profile', '?')}")
        if meta.get("description"):
            terminalreporter.write_line(meta["description"])
        if meta.get("model"):
            terminalreporter.write_line(f"Model: {meta['model']}")
        terminalreporter.write_line("")

        phase_keys = sorted(k for k in breakdown if k.startswith(("1_", "2_", "3_", "4_")))
        phase_sum = sum(breakdown[k] for k in phase_keys)
        total = breakdown.get("0_total_wall", phase_sum)

        terminalreporter.write_line(f"{'Phase':<28} {'Seconds':>14} {'Share':>8}")
        terminalreporter.write_line("-" * 54)
        for key in phase_keys:
            label = key.split("_", 1)[1].replace("_", " ")
            seconds = breakdown[key]
            share = 100.0 * seconds / total if total > 0 else 0.0
            terminalreporter.write_line(
                f"{label:<28} {_format_seconds(seconds):>14} {share:7.1f}%"
            )
        terminalreporter.write_line("-" * 54)
        terminalreporter.write_line(
            f"{'sum of phases':<28} {_format_seconds(phase_sum):>14} "
            f"{100.0 * phase_sum / total if total > 0 else 0.0:7.1f}%"
        )
        overhead = total - phase_sum
        terminalreporter.write_line(
            f"{'measurement overhead':<28} {_format_seconds(overhead):>14} "
            f"{100.0 * overhead / total if total > 0 else 0.0:7.1f}%"
        )
        terminalreporter.write_line(
            f"{'total wall clock':<28} {_format_seconds(total):>14} {100.0:7.1f}%"
        )

        if meta.get("throughput"):
            terminalreporter.write_line("")
            terminalreporter.write_line(f"Throughput: {meta['throughput']}")
        terminalreporter.write_line("")
