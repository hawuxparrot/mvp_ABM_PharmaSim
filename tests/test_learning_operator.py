from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from learning.models import AlertCase
from learning.operator import HiddenOperator


RULES = Path("configs/hitl/operator_rules.v1.json")


def _case(**overrides: object) -> AlertCase:
    base = AlertCase(
        case_id="case",
        tick=1,
        pack_id=1,
        event_type="MOVE",
        alert_code=None,
        decommission_reason=None,
        from_location_id=1,
        to_location_id=2,
        from_role="PHARMACY",
        to_role="PHARMACY",
        from_market="BG",
        to_market="BG",
        same_location=False,
        same_organization=False,
        same_city=False,
        same_postal_code=False,
        distance_km=10.0,
        market_mismatch=False,
        prior_pack_events=0,
        prior_pack_alerts=0,
        source_recent_event_count=0,
        destination_recent_event_count=0,
        source_trust_score=1.0,
        destination_trust_score=1.0,
    )
    return replace(base, **overrides)


def test_hospital_to_hospital_operator_labeling() -> None:
    operator = HiddenOperator.from_json(RULES)
    feedback = operator.label(_case(from_role="HOSPITAL", to_role="HOSPITAL"))
    assert feedback.label is True
    assert feedback.rule_id == "unsafe_hospital_to_hospital_move"


def test_unsafe_rule_priority_over_safe_rules() -> None:
    operator = HiddenOperator.from_json(RULES)
    feedback = operator.label(
        _case(
            event_type="ALERT",
            alert_code="UNKNOWN_SERIAL",
            same_location=True,
            same_organization=True,
        )
    )
    assert feedback.label is True
    assert feedback.rule_id == "unsafe_unknown_serial"
