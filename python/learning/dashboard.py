"""Supervisor-facing Streamlit dashboard for the HITL alert-learning demo.

Reads a single artifact directory produced by ``scripts/run_hitl_demo.py`` and renders
the supervisor walkthrough: the meeting reconstruction, the Bulgaria network and routes,
the layered alert triage with pack history, the oracle learning curves, and the concrete
EMVO/NMVS data request. It never re-runs the C++ engine.

Launch via ``scripts/run_hitl_dashboard.py`` (which wires ``streamlit run``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from learning.artifact_loader import (
    ArtifactBundle,
    alert_rate_comparison,
    learned_policy_rows,
    learning_curve_rows,
    load_bundle,
    pack_trace_index,
    rule_distribution,
    triage_rows,
)
from learning.data_request import EMVO_DATA_REQUEST


DEFAULT_ARTIFACT_DIR = "artifacts/hitl_demo/supervisor"

ROLE_COLORS = {
    "OBP": "#d62728",
    "MANUFACTURER": "#d62728",
    "WHOLESALER": "#1f77b4",
    "PHARMACY": "#2ca02c",
    "HOSPITAL": "#9467bd",
    "LOCAL_ORG": "#7f7f7f",
    "UNKNOWN": "#bcbcbc",
}


def run(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> None:
    st.set_page_config(page_title="PharmaSim HITL AlertLabeler", layout="wide")
    artifact_dir = st.sidebar.text_input("Artifact directory", value=str(artifact_dir))
    try:
        bundle = load_bundle(artifact_dir)
    except FileNotFoundError:
        st.error(
            f"No artifacts found at `{artifact_dir}`.\n\n"
            "Generate them first:\n\n"
            "```bash\nuv run python scripts/run_hitl_demo.py "
            "--query-policy all --mode oracle "
            "--output artifacts/hitl_demo/supervisor\n```"
        )
        return

    st.sidebar.markdown(f"**commit** `{bundle.commit_hash}`")
    _render_sidebar_summary(bundle)

    st.title("PharmaSim HITL AlertLabeler")
    st.caption(
        "A layered NMVS alert-triage workstation that learns the operator's policy from "
        "labels alone. Built to be usable now and to make the EMVO data ask concrete."
    )

    overview, network, triage, learning, data_ask = st.tabs(
        [
            "Overview",
            "Network & routes",
            "Layered triage",
            "Learning",
            "EMVO data request",
        ]
    )
    with overview:
        _render_overview(bundle)
    with network:
        _render_network(bundle)
    with triage:
        _render_triage(bundle)
    with learning:
        _render_learning(bundle)
    with data_ask:
        _render_data_request(bundle)


def _render_sidebar_summary(bundle: ArtifactBundle) -> None:
    final = bundle.metrics.get("final", {})
    counts = bundle.metrics.get("class_counts", {})
    if final:
        st.sidebar.metric("Final F1", f"{float(final.get('f1', 0.0)):.3f}")
        st.sidebar.metric("False positive rate", f"{float(final.get('false_positive_rate', 0.0)):.3f}")
        st.sidebar.metric("Operator labels used", int(final.get("labels_requested", 0)))
    if counts:
        st.sidebar.write(
            f"Cases: {counts.get('total', 0)} "
            f"(unsafe {counts.get('unsafe', 0)} / safe {counts.get('safe', 0)})"
        )


def _render_overview(bundle: ArtifactBundle) -> None:
    st.header("What the supervisor meeting decided")
    st.markdown(
        "This is not 'add ML to the simulator'. It is a **layered alert-triage product** "
        "we can demonstrate today and then calibrate with real EMVO data."
    )
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        "**Layer 1 - system alerts**\n\n"
        "The C++ engine emits real EMVS-typed alerts (`UNKNOWN_SERIAL`, `OUT_OF_MARKET`, "
        "`DUPLICATE_SERIAL`, `STOLEN` decommissions) plus suspicious moves."
    )
    col2.markdown(
        "**Layer 2 - rule database**\n\n"
        "An operator's rules filter and label the queue: same postcode / same pharmacy "
        "-> safe; hospital -> hospital -> alert; low trust -> alert."
    )
    col3.markdown(
        "**Layer 3 - pack history**\n\n"
        "Each case carries the pack's prior events and alerts, and its full location "
        "trace, so a reviewer can see how it got here."
    )

    st.subheader("BDA examples encoded as rules")
    rules = _load_operator_rules(bundle)
    if rules:
        st.dataframe(
            [
                {
                    "rule_id": rule.get("id"),
                    "verdict": "unsafe" if rule.get("label") else "safe",
                    "confidence": rule.get("confidence"),
                    "condition": json.dumps(rule.get("when", {}), sort_keys=True),
                }
                for rule in rules
            ],
            width="stretch",
            hide_index=True,
        )
    st.info(
        "The learner never reads this rule file or the perturbation labels. It only sees "
        "operator yes/no feedback, and must converge to the same policy - the core claim "
        "of the demo."
    )

    st.subheader("What we can show today vs what EMVO data unlocks")
    st.markdown(
        "- **Today:** end-to-end triage, learning convergence, map, routes, pack history.\n"
        "- **Deferred until data arrives:** in-browser human labeling, live tick loop, a "
        "runtime pattern database, and a dynamic blacklist/trust debuff.\n"
        "- **Why it matters:** see the *EMVO data request* tab - we cannot claim the "
        "0.05% target or the 3000/week volume without anonymized NMVS aggregates."
    )


def _render_network(bundle: ArtifactBundle) -> None:
    st.header("Bulgaria distribution network")
    network = bundle.network
    if not network.get("nodes"):
        st.warning("No network data in this artifact directory.")
        return
    if not network.get("geocoded", False):
        st.warning(
            "This run was not geocoded, so nodes have no coordinates. Re-run with "
            "geocoding enabled (drop `--no-geocode`) for the map."
        )

    role_counts = network.get("role_counts", {})
    st.write(
        "Nodes by role: "
        + ", ".join(f"{role} ({count})" for role, count in sorted(role_counts.items()))
    )

    show_edges = st.checkbox("Show supply edges", value=True)
    fig = _network_figure(network, show_edges=show_edges)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Marker ring = learner flagged unsafe; X = injected fraud target. Edges are a "
        "k-NN haversine proxy, not licensed supply links - that is request #1 in the "
        "EMVO data tab."
    )


def _network_figure(network: dict[str, Any], *, show_edges: bool) -> Any:
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    nodes = [n for n in network["nodes"] if n.get("latitude") is not None and n.get("longitude") is not None]
    if not nodes:
        return None
    coord_by_id = {n["location_id"]: (n["longitude"], n["latitude"]) for n in nodes}

    fig = go.Figure()
    if show_edges:
        edge_x: list[Any] = []
        edge_y: list[Any] = []
        for edge in network.get("edges", []):
            a = coord_by_id.get(edge["src_location_id"])
            b = coord_by_id.get(edge["dst_location_id"])
            if a is None or b is None:
                continue
            edge_x.extend([a[0], b[0], None])
            edge_y.extend([a[1], b[1], None])
        if edge_x:
            fig.add_trace(
                go.Scattergl(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line={"width": 0.5, "color": "rgba(120,120,120,0.35)"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    by_role: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_role.setdefault(node.get("role", "UNKNOWN"), []).append(node)
    for role, role_nodes in sorted(by_role.items()):
        fig.add_trace(
            go.Scattergl(
                x=[n["longitude"] for n in role_nodes],
                y=[n["latitude"] for n in role_nodes],
                mode="markers",
                name=f"{role} ({len(role_nodes)})",
                marker={"size": 7, "color": ROLE_COLORS.get(role, "#bcbcbc")},
                text=[
                    f"{n['ext_id']}<br>{role}<br>{n.get('city') or ''} "
                    f"{n.get('postal_code') or ''}<br>trust={n.get('trust_score')}"
                    f"<br>alerts={n.get('alert_count', 0)}"
                    for n in role_nodes
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    flagged = [n for n in nodes if n.get("predicted_unsafe")]
    if flagged:
        fig.add_trace(
            go.Scattergl(
                x=[n["longitude"] for n in flagged],
                y=[n["latitude"] for n in flagged],
                mode="markers",
                name="learner flagged",
                marker={"size": 14, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#ff7f0e"}},
                hoverinfo="skip",
            )
        )
    targets = [n for n in nodes if n.get("target")]
    if targets:
        fig.add_trace(
            go.Scattergl(
                x=[n["longitude"] for n in targets],
                y=[n["latitude"] for n in targets],
                mode="markers",
                name="injected fraud target",
                marker={"size": 12, "color": "#d62728", "symbol": "x"},
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=650,
        xaxis_title="longitude",
        yaxis_title="latitude",
        legend={"orientation": "h"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


def _render_triage(bundle: ArtifactBundle) -> None:
    st.header("Layered alert triage")
    rows = triage_rows(bundle)
    if not rows:
        st.warning("No operator feedback recorded in this artifact directory.")
        return

    dist = rule_distribution(bundle)
    st.subheader("Which rule filtered each alert")
    st.dataframe(dist, width="stretch", hide_index=True)

    st.subheader("Alert queue (as the operator saw it)")
    only_unsafe = st.checkbox("Only alerts kept as unsafe", value=False)
    view = [r for r in rows if r["operator_label"]] if only_unsafe else rows
    st.dataframe(view[:500], width="stretch", hide_index=True)
    st.caption(
        "Each row shows Layer 1 (the case), Layer 2 (`operator_rule_id`, the database "
        "rule that decided it) and the learner's probability. Example: a pharmacy->"
        "pharmacy move in one postcode is filtered safe; a hospital->hospital move stays "
        "an alert."
    )

    st.subheader("Layer 3 - pack history")
    traces = pack_trace_index(bundle)
    if not traces:
        st.info("No pack traces available for this run.")
        return
    pack_id = st.selectbox("Pack", options=sorted(traces.keys()))
    trace = traces[int(pack_id)]
    st.write("Location trace (initial + after each move):")
    st.dataframe(trace.get("trace", []), width="stretch", hide_index=True)
    st.write("All events for this pack:")
    st.dataframe(trace.get("events", []), width="stretch", hide_index=True)


def _render_learning(bundle: ArtifactBundle) -> None:
    st.header("The learner converges to the operator")
    rows = learning_curve_rows(bundle)
    if not rows:
        st.warning("No learning history in this artifact directory.")
        return
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[r["labels_requested"] for r in rows],
                y=[r["f1"] for r in rows],
                mode="lines+markers",
                name="F1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[r["labels_requested"] for r in rows],
                y=[r["false_positive_rate"] for r in rows],
                mode="lines+markers",
                name="False positive rate",
            )
        )
        fig.update_layout(
            height=430,
            xaxis_title="operator labels requested",
            yaxis_title="score",
            yaxis={"range": [-0.02, 1.02]},
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.line_chart(
            {"F1": [r["f1"] for r in rows], "FPR": [r["false_positive_rate"] for r in rows]}
        )

    final = bundle.metrics.get("final", {})
    acceptance = bundle.metrics.get("acceptance_status", {})
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Final metrics")
        st.json(
            {
                k: final.get(k)
                for k in ("f1", "precision", "recall", "false_positive_rate", "labels_requested", "query_rate")
            }
        )
    with col2:
        st.subheader("Acceptance checks")
        st.json(acceptance)

    if bundle.acceptance:
        st.subheader("Five-seed, two-policy acceptance")
        passed = bundle.acceptance.get("passed")
        if passed is True:
            st.success("All acceptance thresholds passed (query-all and uncertainty).")
        elif passed is False:
            st.warning("Some acceptance thresholds failed - see details below.")
        st.json(bundle.acceptance)

    st.subheader("Learned policy (top feature weights)")
    st.caption(
        "The model started predicting nothing and recovered the operator's rules from "
        "feedback alone. Positive weights push toward 'unsafe'."
    )
    st.dataframe(learned_policy_rows(bundle), width="stretch", hide_index=True)


def _render_data_request(bundle: ArtifactBundle) -> None:
    st.header("EMVO / NMVS data request")
    st.markdown(
        "Why we need real data: the simulation currently uses synthetic scan rates and a "
        "k-NN proxy network, so we cannot yet claim the meeting's operational numbers. "
        "Each row below turns a meeting figure into a concrete, anonymized ask."
    )

    comparison = alert_rate_comparison(bundle)
    if comparison.get("sim_alert_rate") is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Simulation alert rate", f"{comparison['sim_alert_rate'] * 100:.3f}%")
        target = comparison.get("target_alert_rate")
        col2.metric("Target alert rate", f"{(target or 0) * 100:.3f}%")
        col3.metric(
            "Sim alerts/week (baseline)",
            f"{comparison.get('sim_alerts_per_week') or 0:.0f}",
            help="Bulgaria reference: ~3000/week",
        )
        st.warning(
            "The simulated alert **rate** (~86%) is far above the real <0.5%, because scan "
            "rates are synthetic. As a result the case stream is heavily unsafe-dominated, "
            "which keeps the held-out false-positive rate high (~0.25) even though the "
            "learner recovers the operator policy (F1 ~0.99). Closing this gap is the whole "
            "point of the data request: with real verify/decommission rates, the alert rate "
            "becomes realistic and the FPR target becomes reachable."
        )

    _render_class_balance(bundle)

    st.dataframe(
        [
            {
                "priority": item.priority,
                "topic": item.topic,
                "meeting signal": item.meeting_signal,
                "simulation today": item.simulation_today,
                "engine parameter": item.engine_parameter,
                "requested aggregate": item.requested_aggregate,
            }
            for item in EMVO_DATA_REQUEST
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "This table is also written to `docs/emvo_data_request.md` so it can leave the "
        "room as a one-page ask."
    )


def _render_class_balance(bundle: ArtifactBundle) -> None:
    counts = bundle.metrics.get("class_counts", {})
    total = int(counts.get("total", 0))
    if total <= 0:
        return
    unsafe = int(counts.get("unsafe", 0))
    safe = int(counts.get("safe", 0))
    st.subheader("Case-stream class balance")
    st.write(
        f"Operator-labeled cases: **{total:,}** "
        f"(unsafe {unsafe:,} = {unsafe / total * 100:.1f}%, safe {safe:,} = "
        f"{safe / total * 100:.1f}%)."
    )
    st.caption(
        "The imbalance is a direct consequence of the synthetic alert rate above - not a "
        "modeling choice. Real NMVS scan statistics would rebalance it."
    )


def _load_operator_rules(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    operator_path = bundle.config.get("config", {}).get("operator_path")
    candidates: list[Path] = []
    if operator_path:
        candidates.append(Path(operator_path))
    candidates.append(Path("configs/hitl/operator_rules.v1.json"))
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rules = data.get("rules")
            if isinstance(rules, list):
                return rules
    return []


if __name__ == "__main__":
    import sys

    run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARTIFACT_DIR)
