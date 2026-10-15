from __future__ import annotations

import json
from types import SimpleNamespace

from canonical.models import ActorRole, CanonicalDataset, CanonicalNode
from compiler.compile import compile_scenario
from learning.artifact_loader import (
    alert_rate_comparison,
    learning_curve_rows,
    load_bundle,
    pack_trace_index,
    rule_distribution,
    triage_rows,
)
from learning.artifacts import (
    build_calibration_payload,
    build_network_payload,
    build_pack_traces,
    write_supervisor_artifacts,
)
from learning.case_builder import CaseBuilderConfig
from learning.runner import CaseStream, HitlDemoConfig
from policy.models import (
    Batch,
    Location,
    LocationEdge,
    Organization,
    OrgType,
    Pack,
    PackState,
    Product,
    ProductCode,
    ProductCodeScheme,
    Scenario,
)


def _fixture_scenario() -> tuple[Scenario, CanonicalDataset]:
    orgs = [
        Organization("org_obp", OrgType.OBP),
        Organization("org_wh", OrgType.WHOLESALER),
        Organization("org_ph", OrgType.LOCAL_ORG),
    ]
    locs = [
        Location("loc_obp", "org_obp", "BG", "1000"),
        Location("loc_wh", "org_wh", "BG", "1000"),
        Location("loc_ph", "org_ph", "BG", "1000"),
    ]
    product = Product("prod", (ProductCode(ProductCodeScheme.GTIN, "00000000000001", True),))
    batch = Batch("batch", "prod", "org_obp", ("BG",))
    packs = [Pack("pack_0", "prod", "batch", "SN0000", "BG", "loc_obp", PackState.ACTIVE)]
    scenario = Scenario(
        organizations=orgs,
        locations=locs,
        products=[product],
        batches=[batch],
        packs=packs,
        location_edges=[
            LocationEdge("loc_obp", "loc_wh", 1.0, 10),
            LocationEdge("loc_wh", "loc_ph", 1.0, 10),
        ],
        seed=7,
    )
    canonical = CanonicalDataset(
        nodes=[
            CanonicalNode("obp", "OBP", ActorRole.WHOLESALER, "BG", "Sofia", "", "1000", True, 42.0, 23.0),
            CanonicalNode("wh", "WH", ActorRole.WHOLESALER, "BG", "Sofia", "", "1000", True, 42.1, 23.1),
            CanonicalNode("ph", "PH", ActorRole.PHARMACY, "BG", "Plovdiv", "", "2000", True, 42.2, 24.7),
        ],
        edges=[],
    )
    return scenario, canonical


def _fixture_stream() -> CaseStream:
    scenario, canonical = _fixture_scenario()
    inp = compile_scenario(scenario)
    ids = {name: inp.location_ext_id.index(f"loc_{name}") for name in ("obp", "wh", "ph")}
    records = [
        {
            "tick": 1, "pack_id": 0, "event_type": "MOVE",
            "from_location_id": ids["obp"], "to_location_id": ids["wh"],
            "from_location": "loc_obp", "to_location": "loc_wh",
            "alert_code": None, "decommission_reason": None, "market": "BG",
        },
        {
            "tick": 2, "pack_id": 0, "event_type": "VERIFY",
            "from_location_id": None, "to_location_id": ids["ph"],
            "from_location": "—", "to_location": "loc_ph",
            "alert_code": None, "decommission_reason": None, "market": "BG",
        },
        {
            "tick": 2, "pack_id": 0, "event_type": "ALERT",
            "from_location_id": None, "to_location_id": ids["ph"],
            "from_location": "—", "to_location": "loc_ph",
            "alert_code": "UNKNOWN_SERIAL", "decommission_reason": None, "market": "BG",
        },
    ]
    from learning.case_builder import CaseBuilder, build_cases_from_records

    builder = CaseBuilder(inp, scenario, canonical, config=CaseBuilderConfig(safe_sample_rate=1.0))
    cases = build_cases_from_records(records, builder, stream_id="seed1:baseline")
    return CaseStream(
        stream_id="seed1:baseline",
        split="evaluation",
        cases=cases,
        records=records,
        engine_input=inp,
        scenario=scenario,
        canonical=canonical,
        perturbation_type=None,
        perturbation_labels=[],
        seed=1,
    )


def test_calibration_counts_scans_and_alerts_and_dedupes_shared_records() -> None:
    stream = _fixture_stream()
    # Single-seed runs share the same records list between the train/eval halves.
    payload = build_calibration_payload(
        [stream],
        [stream],
        commit_hash="abc123",
        ticks=14,
        builder_config=CaseBuilderConfig(),
    )
    assert len(payload["per_stream"]) == 1  # deduped on record identity
    baseline = payload["simulation"]["baseline"]
    assert baseline["scans"] == 1  # one VERIFY (MOVE and ALERT are not scans)
    assert baseline["alerts"] == 1
    assert baseline["alerts_per_week"] == 1 / 14 * 7
    assert payload["reference"]["bulgaria_alerts_per_week"] == 3000
    sales = payload["simulation"]["sales_benchmark"]
    assert sales["n_packs_delivered"] == 1
    assert sales["fraction_within_horizon"] == 1.0


def test_pack_traces_include_events_and_location_trace() -> None:
    stream = _fixture_stream()
    traces = build_pack_traces(stream, commit_hash="abc123")
    assert len(traces) == 1
    trace = traces[0]
    assert trace["pack_id"] == 0
    assert [e["event_type"] for e in trace["events"]] == ["MOVE", "VERIFY", "ALERT"]
    assert trace["trace"][0]["location_ext_id"] == "loc_obp"  # initial location
    assert any(step["location_ext_id"] == "loc_wh" for step in trace["trace"])


def test_pack_traces_with_fake_engine_input() -> None:
    inp = SimpleNamespace(
        n_packs=2,
        pack_initial_location_id=[0, 1],
        pack_ext_id=["pack_0", "pack_1"],
        location_ext_id=["loc_a", "loc_b"],
    )
    records = [
        {"tick": 1, "pack_id": 0, "event_type": "MOVE", "from_location_id": 0, "to_location_id": 1,
         "from_location": "loc_a", "to_location": "loc_b", "alert_code": None,
         "decommission_reason": None, "market": "BG"},
    ]
    from learning.models import AlertCase

    case = AlertCase(
        case_id="s:t1:e0:MOVE:p0:0:1", tick=1, pack_id=0, event_type="MOVE",
        alert_code=None, decommission_reason=None, from_location_id=0, to_location_id=1,
        from_role="PHARMACY", to_role="PHARMACY", from_market="BG", to_market="BG",
        same_location=False, same_organization=False, same_city=False, same_postal_code=False,
        distance_km=1.0, market_mismatch=False, prior_pack_events=0, prior_pack_alerts=0,
        source_recent_event_count=0, destination_recent_event_count=0,
        source_trust_score=1.0, destination_trust_score=1.0,
    )
    stream = CaseStream("s", "evaluation", [case], records, engine_input=inp, seed=1)
    traces = build_pack_traces(stream, commit_hash="x")
    assert traces[0]["case_ids"] == ["s:t1:e0:MOVE:p0:0:1"]
    assert traces[0]["trace"][0]["location_ext_id"] == "loc_a"


def test_network_payload_has_nodes_and_edges() -> None:
    stream = _fixture_stream()
    payload = build_network_payload(
        stream,
        commit_hash="abc123",
        builder_config=CaseBuilderConfig(),
        predicted_location_ids={0},
        target_location_ids={2},
    )
    assert payload["geocoded"] is True
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 2
    flagged = [n for n in payload["nodes"] if n["predicted_unsafe"]]
    assert len(flagged) == 1
    roles = payload["role_counts"]
    assert roles.get("PHARMACY", 0) >= 1


def test_network_payload_handles_missing_stream() -> None:
    payload = build_network_payload(
        None,
        commit_hash="x",
        builder_config=CaseBuilderConfig(),
        predicted_location_ids=set(),
        target_location_ids=set(),
    )
    assert payload["nodes"] == []
    assert payload["geocoded"] is False


def test_write_and_load_round_trip(tmp_path) -> None:
    stream = _fixture_stream()
    config = HitlDemoConfig(
        ticks=14, seeds=(1,), perturbations=(), output_dir=tmp_path,
        safe_sample_rate=1.0, geocode=False, progress=False,
    )
    # Minimal companion files so the loader has something to read.
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "commit_hash": "abc123",
                "class_counts": {"safe": 1, "unsafe": 2, "total": 3},
                "final": {"f1": 0.9, "false_positive_rate": 0.05, "labels_requested": 2},
                "acceptance_status": {"final_f1_at_least_0_85": True},
                "metrics_history": [
                    {"labels_requested": 0, "cases_processed": 0, "f1": 0.0, "false_positive_rate": 0.0},
                    {"labels_requested": 2, "cases_processed": 2, "f1": 0.9, "false_positive_rate": 0.05},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "feedback.jsonl").write_text(
        json.dumps(
            {"case_id": "c1", "stream_id": "seed1:baseline", "tick": 2, "probability": 0.8,
             "prediction": True, "operator_label": True, "operator_rule_id": "unsafe_unknown_serial"}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "feature_names.json").write_text(
        json.dumps({"feature_names": ["bias", "same_postal_code"]}), encoding="utf-8"
    )
    (tmp_path / "model_weights.json").write_text(
        json.dumps({"weights": [-1.0, 0.5]}), encoding="utf-8"
    )

    write_supervisor_artifacts(
        tmp_path,
        train_streams=[stream],
        eval_streams=[stream],
        map_stream=stream,
        config=config,
        commit_hash="abc123",
        predicted_location_ids={0},
        target_location_ids={2},
    )
    assert (tmp_path / "network.json").exists()
    assert (tmp_path / "pack_traces.jsonl").exists()
    assert (tmp_path / "calibration.json").exists()

    bundle = load_bundle(tmp_path)
    assert bundle.commit_hash == "abc123"
    assert bundle.has_network()
    assert bundle.has_learning()
    assert learning_curve_rows(bundle)[-1]["f1"] == 0.9
    assert triage_rows(bundle)[0]["operator_rule_id"] == "unsafe_unknown_serial"
    assert rule_distribution(bundle)[0]["rule_id"] == "unsafe_unknown_serial"
    assert 0 in pack_trace_index(bundle)
    assert alert_rate_comparison(bundle)["bulgaria_alerts_per_week"] == 3000
