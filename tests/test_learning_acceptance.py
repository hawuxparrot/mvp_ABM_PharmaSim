from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from learning.acceptance import run_seed_holdout_acceptance
from learning.models import AlertCase
from learning.operator import HiddenOperator
from learning.runner import CaseStream, HitlDemoConfig


RULES = Path("configs/hitl/operator_rules.v1.json")


def _case(case_id: str, *, unsafe: bool) -> AlertCase:
    base = AlertCase(
        case_id=case_id,
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
        same_postal_code=True,
        distance_km=2.0,
        market_mismatch=False,
        prior_pack_events=0,
        prior_pack_alerts=0,
        source_recent_event_count=0,
        destination_recent_event_count=0,
        source_trust_score=1.0,
        destination_trust_score=1.0,
    )
    if unsafe:
        return replace(base, event_type="ALERT", alert_code="UNKNOWN_SERIAL")
    return base


def test_acceptance_suite_reports_both_policies_and_all_seed_holdouts() -> None:
    streams = [
        CaseStream(
            stream_id=f"seed{seed}",
            split="all",
            cases=[
                _case(f"seed{seed}-unsafe", unsafe=True),
                _case(f"seed{seed}-safe", unsafe=False),
            ],
            records=[],
            seed=seed,
        )
        for seed in range(1, 6)
    ]
    result = run_seed_holdout_acceptance(
        streams,
        HiddenOperator.from_json(RULES),
        config=HitlDemoConfig(seeds=(1, 2, 3, 4, 5), progress=False),
    )
    assert set(result["policies"]) == {"all", "uncertainty"}
    assert set(result["policies"]["all"]["per_seed"]) == {"1", "2", "3", "4", "5"}
    assert set(result["acceptance_status"]) == {
        "case_stream_has_safe_and_unsafe",
        "query_all_final_f1_at_least_0_85",
        "query_all_final_fpr_at_most_0_10",
        "query_all_f1_improved_by_0_20",
        "at_least_four_of_five_seed_holdouts_meet_f1",
        "uncertainty_uses_at_most_half_the_labels",
        "uncertainty_f1_within_0_05_of_query_all",
    }
