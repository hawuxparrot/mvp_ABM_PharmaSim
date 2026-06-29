"""Smoke tests for geographic graph + flow-comparison plots (no native engine needed)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from canonical.pipeline import build_bulgaria_scenario  # noqa: E402
from pathlib import Path  # noqa: E402

from runtime.simulation_viz import plot_flow_comparison, plot_location_graph  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_plot_location_graph_smoke() -> None:
    scenario, canonical = build_bulgaria_scenario(
        bg_registry_path=str(REPO / "data" / "data.json"),
        spor_locations_path=str(REPO / "data" / "spor_locations.csv"),
        geocode=False,
        max_locations=120,
    )
    flagged = [scenario.locations[0].ext_id] if scenario.locations else []
    fig, ax = plot_location_graph(
        canonical, scenario, title="test graph", highlight_location_ext_ids=flagged
    )
    # At least the node scatter groups + legend should have been drawn.
    assert ax.collections
    assert ax.get_title() == "test graph"


def test_plot_flow_comparison_smoke() -> None:
    base = [
        {"event_type": "MOVE"},
        {"event_type": "MOVE"},
        {"event_type": "DECOMMISSION"},
    ]
    pert = [
        {"event_type": "MOVE"},
        {"event_type": "MOVE"},
        {"event_type": "MOVE"},
        {"event_type": "DECOMMISSION"},
        {"event_type": "DECOMMISSION"},
    ]
    fig, ax = plot_flow_comparison(base, pert, title="flow cmp")
    assert ax.get_title() == "flow cmp"
    assert len(ax.patches) == 4  # 2 event types x 2 series
