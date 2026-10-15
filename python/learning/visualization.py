from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def write_hitl_plots(
    output_dir: Path,
    *,
    metrics_history: Sequence[Mapping[str, Any]],
    learning_steps: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    weights: Sequence[float],
    commit_hash: str = "unknown",
    scenario: Any = None,
    canonical: Any = None,
    predicted_location_ids: set[int] | None = None,
    target_location_ids: set[int] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_learning_curve(metrics_history, output_dir / "learning_curve.png", commit_hash)
    _plot_oracle_convergence(
        metrics_history,
        output_dir / "oracle_convergence.png",
        commit_hash,
    )
    _plot_false_positive_rate(metrics_history, output_dir / "false_positive_rate.png", commit_hash)
    _plot_query_efficiency(metrics_history, output_dir / "query_efficiency.png", commit_hash)
    _plot_model_weights(feature_names, weights, output_dir / "model_weights.png", commit_hash)
    _plot_alert_timeline(learning_steps, output_dir / "alert_timeline.png", commit_hash)
    _plot_simulation_map(
        scenario,
        canonical,
        output_dir / "simulation_map.png",
        commit_hash=commit_hash,
        predicted_location_ids=predicted_location_ids or set(),
        target_location_ids=target_location_ids or set(),
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_learning_curve(
    metrics_history: Sequence[Mapping[str, Any]], path: Path, commit_hash: str
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    if metrics_history:
        x = [int(row["labels_requested"]) for row in metrics_history]
        ax.plot(x, [float(row["f1"]) for row in metrics_history], marker="o", label="F1")
        ax.plot(
            x,
            [float(row["false_positive_rate"]) for row in metrics_history],
            marker="s",
            label="FPR",
        )
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No metric points", ha="center", va="center")
    ax.set_xlabel("operator labels")
    ax.set_ylabel("score")
    ax.set_title("Learning curve")
    ax.set_ylim(-0.02, 1.02)
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_false_positive_rate(
    metrics_history: Sequence[Mapping[str, Any]], path: Path, commit_hash: str
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if metrics_history:
        ax.plot(
            [int(row["labels_requested"]) for row in metrics_history],
            [float(row["false_positive_rate"]) for row in metrics_history],
            marker="o",
        )
    else:
        ax.text(0.5, 0.5, "No metric points", ha="center", va="center")
    ax.set_xlabel("operator labels")
    ax.set_ylabel("false positive rate")
    ax.set_title("False positive rate")
    ax.set_ylim(-0.02, 1.02)
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_oracle_convergence(
    metrics_history: Sequence[Mapping[str, Any]],
    path: Path,
    commit_hash: str,
) -> None:
    plt = _pyplot()
    fig, (rate_ax, error_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    if metrics_history:
        x = [int(row["labels_requested"]) for row in metrics_history]
        rate_ax.plot(
            x,
            [float(row["oracle_unsafe_rate"]) for row in metrics_history],
            color="black",
            linestyle="--",
            label="oracle unsafe rate",
        )
        rate_ax.plot(
            x,
            [float(row["learned_unsafe_rate"]) for row in metrics_history],
            color="tab:blue",
            label="learned unsafe rate",
        )
        error_ax.plot(
            x,
            [float(row["disagreement_rate"]) for row in metrics_history],
            color="tab:red",
            label="decision disagreement",
        )
        error_ax.plot(
            x,
            [float(row["probability_mae"]) for row in metrics_history],
            color="tab:orange",
            label="probability error",
        )
        rate_ax.legend(loc="best")
        error_ax.legend(loc="best")
    else:
        rate_ax.text(0.5, 0.5, "No metric points", ha="center", va="center")
        error_ax.text(0.5, 0.5, "No metric points", ha="center", va="center")
    rate_ax.set_ylabel("unsafe share")
    rate_ax.set_title("Oracle and learner convergence")
    rate_ax.set_ylim(-0.02, 1.02)
    error_ax.set_xlabel("operator labels")
    error_ax.set_ylabel("error")
    error_ax.set_ylim(-0.02, 1.02)
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_query_efficiency(
    metrics_history: Sequence[Mapping[str, Any]], path: Path, commit_hash: str
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if metrics_history:
        ax.plot(
            [int(row["labels_requested"]) for row in metrics_history],
            [float(row["f1"]) for row in metrics_history],
            marker="o",
        )
    else:
        ax.text(0.5, 0.5, "No metric points", ha="center", va="center")
    ax.set_xlabel("requested labels")
    ax.set_ylabel("evaluation F1")
    ax.set_title("Query efficiency")
    ax.set_ylim(-0.02, 1.02)
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_model_weights(
    feature_names: Sequence[str],
    weights: Sequence[float],
    path: Path,
    commit_hash: str,
) -> None:
    plt = _pyplot()
    values = np.asarray(weights, dtype=float)
    names = list(feature_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(values) and len(values) == len(names):
        order = np.argsort(np.abs(values))[-20:]
        order = order[np.argsort(values[order])]
        ax.barh([names[i] for i in order], [values[i] for i in order])
    else:
        ax.text(0.5, 0.5, "No weights", ha="center", va="center")
    ax.set_xlabel("weight")
    ax.set_title("Largest model weights")
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_alert_timeline(
    learning_steps: Sequence[Mapping[str, Any]], path: Path, commit_hash: str
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if learning_steps:
        pred_x: list[int] = []
        pred_c: list[str] = []
        label_x: list[int] = []
        label_c: list[str] = []
        for row in learning_steps:
            pred_x.append(int(row["tick"]))
            pred_c.append("tab:red" if _truthy(row.get("prediction")) else "tab:blue")
            if _truthy(row.get("queried")) and row.get("operator_label") != "":
                label_x.append(int(row["tick"]))
                label_c.append("tab:red" if _truthy(row.get("operator_label")) else "tab:blue")
        ax.scatter(pred_x, [0] * len(pred_x), c=pred_c, s=16, label="learner")
        if label_x:
            ax.scatter(label_x, [1] * len(label_x), c=label_c, s=20, marker="s", label="operator")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No learning steps", ha="center", va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["learner", "operator"])
    ax.set_xlabel("tick")
    ax.set_title("Alert timeline")
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_simulation_map(
    scenario: Any,
    canonical: Any,
    path: Path,
    *,
    commit_hash: str,
    predicted_location_ids: set[int],
    target_location_ids: set[int],
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 7))
    if scenario is None or canonical is None:
        ax.text(0.5, 0.5, "No map context", ha="center", va="center")
        _finish_map(fig, ax, path, commit_hash)
        return

    coord_by_node = {
        n.node_id: (float(n.longitude), float(n.latitude))
        for n in canonical.nodes
        if n.latitude is not None and n.longitude is not None
    }

    def coord_for_loc(loc_id: int) -> tuple[float, float] | None:
        if loc_id < 0 or loc_id >= len(scenario.locations):
            return None
        loc_ext = scenario.locations[loc_id].ext_id
        node_id = loc_ext[len("loc_") :] if loc_ext.startswith("loc_") else loc_ext
        return coord_by_node.get(node_id)

    xs, ys = [], []
    for loc_id in range(len(scenario.locations)):
        coord = coord_for_loc(loc_id)
        if coord is not None:
            xs.append(coord[0])
            ys.append(coord[1])
    if xs:
        ax.scatter(xs, ys, s=8, c="0.65", alpha=0.6, label="locations")
    _scatter_location_set(
        ax,
        predicted_location_ids & target_location_ids,
        coord_for_loc,
        label="true positive",
        color="tab:green",
        marker="o",
    )
    _scatter_location_set(
        ax,
        predicted_location_ids - target_location_ids,
        coord_for_loc,
        label="false positive",
        color="tab:orange",
        marker="x",
    )
    _scatter_location_set(
        ax,
        target_location_ids - predicted_location_ids,
        coord_for_loc,
        label="false negative",
        color="tab:red",
        marker="s",
    )
    if not xs:
        ax.text(0.5, 0.5, "No coordinates", ha="center", va="center")
    _finish_map(fig, ax, path, commit_hash)


def _scatter_location_set(ax: Any, locs: set[int], coord_for_loc: Any, *, label: str, color: str, marker: str) -> None:
    xs, ys = [], []
    for loc_id in sorted(locs):
        coord = coord_for_loc(loc_id)
        if coord is not None:
            xs.append(coord[0])
            ys.append(coord[1])
    if xs:
        ax.scatter(xs, ys, s=80, c=color, marker=marker, label=label)


def _finish_map(fig: Any, ax: Any, path: Path, commit_hash: str) -> None:
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Simulation map")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize="small")
    _add_commit_footer(fig, commit_hash)
    fig.tight_layout()
    fig.savefig(path)
    plt = _pyplot()
    plt.close(fig)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def _add_commit_footer(fig: Any, commit_hash: str) -> None:
    fig.text(0.99, 0.01, f"commit {commit_hash}", ha="right", va="bottom", fontsize=6)
