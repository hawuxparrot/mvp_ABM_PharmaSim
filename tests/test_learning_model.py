from __future__ import annotations

import numpy as np

from learning.models import AlertCase, dataclass_from_dict, to_jsonable
from learning.online_model import LearnerConfig, OnlineLogisticLearner


def test_learner_update_direction() -> None:
    learner = OnlineLogisticLearner(
        ["bias", "x"],
        config=LearnerConfig(learning_rate=0.2, l2_strength=0.0, initial_bias=0.0),
    )
    features = np.asarray([1.0, 1.0])
    before = learner.predict_proba(features)
    learner.update(features, True)
    after = learner.predict_proba(features)
    assert after > before


def test_checkpoint_save_and_load(tmp_path) -> None:
    learner = OnlineLogisticLearner(["bias", "x"])
    learner.update(np.asarray([1.0, 1.0]), True)
    path = tmp_path / "model.json"
    learner.save_json(path)
    loaded = OnlineLogisticLearner.load_json(path)
    assert loaded.feature_names == learner.feature_names
    assert loaded.weights.tolist() == learner.weights.tolist()
    assert loaded.updates == learner.updates


def test_learning_dataclasses_round_trip_json() -> None:
    case = AlertCase(
        case_id="c1",
        tick=1,
        pack_id=1,
        event_type="ALERT",
        alert_code="UNKNOWN_SERIAL",
        decommission_reason=None,
        from_location_id=None,
        to_location_id=2,
        from_role=None,
        to_role="PHARMACY",
        from_market=None,
        to_market="BG",
        same_location=False,
        same_organization=False,
        same_city=False,
        same_postal_code=False,
        distance_km=None,
        market_mismatch=False,
        prior_pack_events=3,
        prior_pack_alerts=1,
        source_recent_event_count=0,
        destination_recent_event_count=2,
        source_trust_score=1.0,
        destination_trust_score=0.9,
    )
    payload = to_jsonable(case)
    assert dataclass_from_dict(AlertCase, payload) == case
