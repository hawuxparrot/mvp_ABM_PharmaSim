from __future__ import annotations

from pathlib import Path

import learning.runner as runner_module
import pytest
from learning.features import FeatureEncoder
from learning.models import AlertCase
from learning.online_model import OnlineLogisticLearner
from learning.operator import HiddenOperator
from learning.runner import (
    CaseStream,
    HitlDemoConfig,
    evaluate_streams,
    run_case_streams,
    split_streams_for_evaluation,
)


RULES = Path("configs/hitl/operator_rules.v1.json")
REQUIRED_FILES = {
    "config_resolved.json",
    "cases.jsonl",
    "feedback.jsonl",
    "learning_steps.csv",
    "metrics.json",
    "model_weights.json",
    "feature_names.json",
    "checkpoint.json",
    "learning_curve.png",
    "oracle_convergence.png",
    "false_positive_rate.png",
    "query_efficiency.png",
    "model_weights.png",
    "alert_timeline.png",
    "simulation_map.png",
}


def _case(case_id: str, *, unsafe: bool, tick: int) -> AlertCase:
    return AlertCase(
        case_id=case_id,
        tick=tick,
        pack_id=1,
        event_type="ALERT" if unsafe else "MOVE",
        alert_code="UNKNOWN_SERIAL" if unsafe else None,
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
        same_postal_code=not unsafe,
        distance_km=2.0,
        market_mismatch=False,
        prior_pack_events=0,
        prior_pack_alerts=0,
        source_recent_event_count=0,
        destination_recent_event_count=0,
        source_trust_score=1.0,
        destination_trust_score=1.0,
    )


def _streams() -> tuple[list[CaseStream], list[CaseStream]]:
    train = CaseStream(
        "synthetic",
        "train",
        [_case("train-unsafe", unsafe=True, tick=1), _case("train-safe", unsafe=False, tick=2)],
        [],
    )
    eval_stream = CaseStream(
        "synthetic",
        "evaluation",
        [_case("eval-unsafe", unsafe=True, tick=1), _case("eval-safe", unsafe=False, tick=2)],
        [],
    )
    return [train], [eval_stream]


def test_evaluation_labels_do_not_update_the_learner() -> None:
    _train, eval_streams = _streams()
    operator = HiddenOperator.from_json(RULES)
    encoder = FeatureEncoder()
    learner = OnlineLogisticLearner(encoder.feature_names)
    before = learner.weights.copy()
    evaluation = evaluate_streams(
        eval_streams,
        operator,
        encoder,
        learner,
        threshold=0.5,
        labels_requested=0,
        cases_processed=0,
    )
    assert learner.weights.tolist() == before.tolist()
    assert evaluation["overall"]["oracle_unsafe_rate"] == 0.5
    assert evaluation["overall"]["learned_unsafe_rate"] == 0.0
    assert evaluation["overall"]["disagreement_rate"] == 0.5
    assert evaluation["overall"]["probability_mae"] == pytest.approx(0.5)


def test_small_synthetic_replay_creates_all_report_files(tmp_path) -> None:
    train, eval_streams = _streams()
    operator = HiddenOperator.from_json(RULES)
    config = HitlDemoConfig(
        ticks=2,
        seeds=(1,),
        perturbations=(),
        operator_path=RULES,
        output_dir=tmp_path,
        metric_interval=1,
        geocode=False,
        progress=False,
    )
    result = run_case_streams(train, eval_streams, operator, config=config)
    assert result.labels_requested == 2
    assert REQUIRED_FILES <= {p.name for p in tmp_path.iterdir()}


def test_deterministic_output_from_fixed_seed(tmp_path) -> None:
    paths = [tmp_path / "run1", tmp_path / "run2"]
    for path in paths:
        train, eval_streams = _streams()
        config = HitlDemoConfig(
            ticks=2,
            seeds=(1,),
            perturbations=(),
            operator_path=RULES,
            output_dir=path,
            metric_interval=1,
            geocode=False,
            progress=False,
        )
        run_case_streams(train, eval_streams, HiddenOperator.from_json(RULES), config=config)
    assert (paths[0] / "learning_steps.csv").read_text() == (paths[1] / "learning_steps.csv").read_text()
    assert (paths[0] / "model_weights.json").read_text() == (paths[1] / "model_weights.json").read_text()


def test_multi_seed_split_holds_out_the_complete_seed() -> None:
    streams = [
        CaseStream(
            f"seed{seed}",
            "all",
            [_case(f"seed{seed}-case", unsafe=seed == 3, tick=seed)],
            [],
            seed=seed,
        )
        for seed in (1, 2, 3)
    ]
    train, evaluation = split_streams_for_evaluation(
        streams,
        seeds=(1, 2, 3),
        evaluation_seed=2,
        evaluation_fraction=0.30,
    )
    assert {stream.seed for stream in train} == {1, 3}
    assert {stream.seed for stream in evaluation} == {2}


def test_unqueried_cases_do_not_repeat_interval_evaluation(tmp_path, monkeypatch) -> None:
    cases = [
        _case(f"train-{index}", unsafe=index == 0, tick=1)
        for index in range(8)
    ]
    train = [CaseStream("train", "train", cases, [])]
    _unused, evaluation = _streams()

    class FirstOnlyPolicy:
        def select_tick(self, rows):
            return {rows[0][0].case_id}

    original_evaluate = runner_module.evaluate_streams
    evaluation_calls = 0

    def counting_evaluate(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner_module, "make_query_policy", lambda *_args, **_kwargs: FirstOnlyPolicy())
    monkeypatch.setattr(runner_module, "evaluate_streams", counting_evaluate)
    run_case_streams(
        train,
        evaluation,
        HiddenOperator.from_json(RULES),
        config=HitlDemoConfig(
            output_dir=tmp_path,
            query_policy="uncertainty",
            metric_interval=1,
            geocode=False,
            progress=False,
        ),
    )
    assert evaluation_calls == 3
