"""Pure readers for a HITL artifact directory.

The dashboard is driven entirely from files on disk. Keeping the loading logic here
(with no Streamlit import) means every tab's data can be unit-tested against a small
fixture directory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArtifactBundle:
    """Everything the dashboard needs, loaded from one output directory."""

    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    network: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    pack_traces: list[dict[str, Any]] = field(default_factory=list)
    acceptance: dict[str, Any] = field(default_factory=dict)

    @property
    def commit_hash(self) -> str:
        for source in (self.metrics, self.config, self.network, self.calibration):
            value = source.get("commit_hash")
            if value:
                return str(value)
        return "unknown"

    def has_learning(self) -> bool:
        return bool(self.metrics.get("metrics_history"))

    def has_network(self) -> bool:
        return bool(self.network.get("nodes"))


def load_bundle(artifact_dir: str | Path) -> ArtifactBundle:
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"artifact directory not found: {path}")
    model = _read_json(path / "model_weights.json") or {}
    feature_payload = _read_json(path / "feature_names.json") or {}
    return ArtifactBundle(
        path=path,
        config=_read_json(path / "config_resolved.json") or {},
        metrics=_read_json(path / "metrics.json") or {},
        network=_read_json(path / "network.json") or {},
        calibration=_read_json(path / "calibration.json") or {},
        feature_names=list(feature_payload.get("feature_names", [])),
        weights=[float(w) for w in model.get("weights", [])],
        feedback=_read_jsonl(path / "feedback.jsonl"),
        pack_traces=_read_jsonl(path / "pack_traces.jsonl"),
        acceptance=_read_json(path / "acceptance.json") or {},
    )


def learning_curve_rows(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    """(labels_requested, f1, false_positive_rate, ...) points for the learning tab."""
    history = bundle.metrics.get("metrics_history", [])
    rows: list[dict[str, Any]] = []
    for row in history:
        rows.append(
            {
                "labels_requested": int(row.get("labels_requested", 0)),
                "cases_processed": int(row.get("cases_processed", 0)),
                "f1": float(row.get("f1", 0.0)),
                "false_positive_rate": float(row.get("false_positive_rate", 0.0)),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
                "query_rate": float(row.get("query_rate", 0.0)),
                "disagreement_rate": float(row.get("disagreement_rate", 0.0)),
            }
        )
    return rows


def learned_policy_rows(bundle: ArtifactBundle, *, top_k: int = 12) -> list[dict[str, Any]]:
    """Largest positive/negative feature weights as a readable learned policy."""
    pairs = list(zip(bundle.feature_names, bundle.weights))
    pairs.sort(key=lambda item: item[1], reverse=True)
    top = pairs[:top_k]
    bottom = [p for p in pairs[-top_k:] if p not in top]
    rows: list[dict[str, Any]] = []
    for name, weight in [*top, *reversed(bottom)]:
        rows.append(
            {
                "feature": name,
                "weight": float(weight),
                "direction": "unsafe" if weight >= 0 else "safe",
            }
        )
    return rows


def triage_rows(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    """One row per labeled case: the three triage layers side by side."""
    rows: list[dict[str, Any]] = []
    for entry in bundle.feedback:
        rule_id = str(entry.get("operator_rule_id", ""))
        rows.append(
            {
                "case_id": entry.get("case_id"),
                "stream_id": entry.get("stream_id"),
                "tick": entry.get("tick"),
                "probability": float(entry.get("probability", 0.0)),
                "prediction": bool(entry.get("prediction", False)),
                "operator_label": bool(entry.get("operator_label", False)),
                "operator_rule_id": rule_id,
                "layer_label": "unsafe" if entry.get("operator_label") else "safe",
            }
        )
    return rows


def rule_distribution(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    """How often each operator rule fired, with its safe/unsafe verdict."""
    counts: dict[str, dict[str, Any]] = {}
    for entry in bundle.feedback:
        rule_id = str(entry.get("operator_rule_id", "unknown")) or "unknown"
        bucket = counts.setdefault(
            rule_id,
            {"rule_id": rule_id, "count": 0, "unsafe": 0, "safe": 0},
        )
        bucket["count"] += 1
        if entry.get("operator_label"):
            bucket["unsafe"] += 1
        else:
            bucket["safe"] += 1
    return sorted(counts.values(), key=lambda b: (-b["count"], b["rule_id"]))


def pack_trace_index(bundle: ArtifactBundle) -> dict[int, dict[str, Any]]:
    return {int(trace["pack_id"]): trace for trace in bundle.pack_traces if "pack_id" in trace}


def alert_rate_comparison(bundle: ArtifactBundle) -> dict[str, Any]:
    """Simulation alert rate / weekly volume next to the meeting reference numbers."""
    calibration = bundle.calibration
    baseline = calibration.get("simulation", {}).get("baseline", {})
    reference = calibration.get("reference", {})
    return {
        "sim_alert_rate": baseline.get("alert_rate"),
        "sim_alerts_per_week": baseline.get("alerts_per_week"),
        "sim_scans": baseline.get("scans"),
        "sim_alerts": baseline.get("alerts"),
        "target_alert_rate": reference.get("target_alert_rate"),
        "current_alert_rate_ceiling": reference.get("current_alert_rate_ceiling"),
        "bulgaria_alerts_per_week": reference.get("bulgaria_alerts_per_week"),
        "germany_alerts_per_week": reference.get("germany_alerts_per_week"),
        "eu_scans_per_year": reference.get("eu_scans_per_year"),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            rows.append(dict(parsed))
    return rows
