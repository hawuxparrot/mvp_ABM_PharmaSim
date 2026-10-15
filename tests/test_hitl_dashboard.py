from __future__ import annotations

import json

import pytest

from learning.data_request import EMVO_DATA_REQUEST, to_markdown_table


def test_data_request_table_is_non_empty_and_renders_markdown() -> None:
    assert len(EMVO_DATA_REQUEST) >= 6
    table = to_markdown_table()
    assert table.startswith("| Priority |")
    # header + separator + one row per item
    assert table.count("\n") == len(EMVO_DATA_REQUEST) + 1
    # every meeting-critical topic is represented
    topics = " ".join(item.topic.lower() for item in EMVO_DATA_REQUEST)
    for keyword in ("verify", "network", "vendor", "blacklist"):
        assert keyword in topics


def test_data_request_doc_matches_table() -> None:
    from pathlib import Path

    doc = Path("docs/emvo_data_request.md")
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for item in EMVO_DATA_REQUEST:
        assert item.topic in text


def test_dashboard_imports_and_reads_fixture(tmp_path) -> None:
    pytest.importorskip("streamlit")
    import learning.dashboard as dashboard
    from learning.artifact_loader import load_bundle, learning_curve_rows

    assert callable(dashboard.run)

    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "commit_hash": "abc",
                "final": {"f1": 0.9, "false_positive_rate": 0.05, "labels_requested": 3},
                "metrics_history": [
                    {"labels_requested": 0, "f1": 0.0, "false_positive_rate": 0.0},
                    {"labels_requested": 3, "f1": 0.9, "false_positive_rate": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "network.json").write_text(
        json.dumps({"commit_hash": "abc", "nodes": [{"location_id": 0, "ext_id": "loc_a", "role": "PHARMACY", "latitude": 42.0, "longitude": 23.0, "trust_score": 1.0}], "edges": [], "geocoded": True, "role_counts": {"PHARMACY": 1}}),
        encoding="utf-8",
    )
    bundle = load_bundle(tmp_path)
    assert bundle.has_network()
    assert learning_curve_rows(bundle)[-1]["f1"] == 0.9
