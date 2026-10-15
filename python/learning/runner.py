from __future__ import annotations

import csv
import heapq
import json
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from compiler.types import EngineInput
from policy.perturbations import (
    PerturbationLabel,
    PerturbationSpec,
    PerturbationType,
    apply_perturbation,
)
from policy.models import Scenario

from .case_builder import CaseBuilder, CaseBuilderConfig, build_cases_from_records
from .features import FeatureEncoder
from .metrics import classification_metrics, target_location_metrics
from .models import AlertCase, OperatorFeedback, to_jsonable
from .online_model import LearnerConfig, OnlineLogisticLearner
from .operator import HiddenOperator
from .query_policy import QueryPolicyConfig, make_query_policy
from .visualization import write_hitl_plots
from .artifacts import write_supervisor_artifacts


_LEARNING_STEP_FIELDS = [
    "commit_hash",
    "stream_id",
    "perturbation_type",
    "case_id",
    "tick",
    "event_type",
    "alert_code",
    "probability",
    "prediction",
    "queried",
    "operator_label",
    "operator_confidence",
    "operator_rule_id",
    "labels_requested",
    "cases_processed",
]


@dataclass(frozen=True)
class HitlDemoConfig:
    scenario: str = "bulgaria"
    ticks: int = 300
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    perturbations: tuple[str, ...] = (
        "COUNTERFEIT",
        "SMUGGLING",
        "THEFT",
        "VOLUME_SPIKE",
    )
    operator_path: Path = Path("configs/hitl/operator_rules.v1.json")
    query_policy: str = "all"
    output_dir: Path = Path("artifacts/hitl_demo")
    mode: str = "oracle"
    resume: bool = False
    learning_rate: float = 0.05
    l2_strength: float = 0.0001
    decision_threshold: float = 0.5
    initial_bias: float = -1.0
    safe_sample_rate: float = 0.10
    safe_sample_seed: int = 1729
    recent_window_ticks: int = 25
    recent_source_candidate_threshold: int = 25
    query_budget_per_tick: int = 8
    uncertainty_margin: float = 0.15
    exploration_rate: float = 0.05
    metric_interval: int = 25
    evaluation_fraction: float = 0.30
    evaluation_seed: int | None = None
    metric_eval_max_cases: int = 2_000
    max_locations: int = 350
    packs_per_obp: int = 60
    packs_per_wholesaler: int = 12
    n_obp_hubs: int = 6
    geocode: bool = True
    geocode_cache_path: Path | None = Path("data/geocode_cache_bg.json")
    geocode_max_new_requests: int | None = 0
    progress: bool = True


@dataclass
class CaseStream:
    stream_id: str
    split: str
    cases: list[AlertCase]
    records: list[dict[str, Any]]
    engine_input: EngineInput | None = None
    scenario: Scenario | None = None
    canonical: Any = None
    perturbation_type: str | None = None
    perturbation_labels: list[PerturbationLabel] | None = None
    seed: int | None = None


@dataclass
class HitlRunResult:
    output_dir: Path
    metrics: dict[str, Any]
    labels_requested: int
    cases_processed: int


class HumanStop(Exception):
    pass


def run_hitl_demo(config: HitlDemoConfig) -> HitlRunResult:
    if config.scenario != "bulgaria":
        raise ValueError("the first HITL demo supports only --scenario bulgaria")
    train_streams, eval_streams, map_stream = build_bulgaria_case_streams(config)
    operator = HiddenOperator.from_json(config.operator_path)
    return run_case_streams(
        train_streams,
        eval_streams,
        operator,
        config=config,
        map_stream=map_stream,
    )


def run_case_streams(
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
    *,
    config: HitlDemoConfig,
    map_stream: CaseStream | None = None,
) -> HitlRunResult:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    commit_hash = repo_commit_hash()
    encoder = FeatureEncoder()
    learner = OnlineLogisticLearner(
        encoder.feature_names,
        config=LearnerConfig(
            learning_rate=config.learning_rate,
            l2_strength=config.l2_strength,
            decision_threshold=config.decision_threshold,
            initial_bias=config.initial_bias,
        ),
    )
    skip_cases = 0
    labels_requested = 0
    checkpoint = output_dir / "checkpoint.json"
    if config.resume and checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        learner = OnlineLogisticLearner.from_dict(payload["learner"])
        skip_cases = int(payload.get("cases_processed", 0))
        labels_requested = int(payload.get("labels_requested", 0))

    _validate_class_counts(train_streams, eval_streams, operator)
    _write_config(
        output_dir,
        config,
        commit_hash,
        encoder.feature_names,
        train_streams=train_streams,
        eval_streams=eval_streams,
    )
    _write_cases(
        output_dir,
        train_streams,
        eval_streams,
        commit_hash,
        progress=config.progress,
    )
    _write_feature_names(output_dir, encoder.feature_names, commit_hash)

    query_policy = make_query_policy(
        config.query_policy,
        QueryPolicyConfig(
            query_budget_per_tick=config.query_budget_per_tick,
            uncertainty_margin=config.uncertainty_margin,
            exploration_rate=config.exploration_rate,
            seed=config.safe_sample_seed + 101,
        ),
    )
    timeline_steps: list[dict[str, Any]] = []
    metrics_history: list[dict[str, Any]] = []
    cases_processed = 0
    last_metric_label_count = labels_requested
    learning_bar: Any = None
    metric_eval_streams = _sample_evaluation_streams(
        eval_streams,
        max_cases=config.metric_eval_max_cases,
    )

    def record_metrics(streams: Sequence[CaseStream]) -> dict[str, Any]:
        evaluation = evaluate_streams(
            streams,
            operator,
            encoder,
            learner,
            threshold=config.decision_threshold,
            labels_requested=labels_requested,
            cases_processed=cases_processed,
        )
        row = {
            **evaluation["overall"],
            "labels_requested": labels_requested,
            "cases_processed": cases_processed,
        }
        metrics_history.append(row)
        if learning_bar is not None:
            learning_bar.set_postfix(
                labels=labels_requested,
                agreement=f"{float(row['f1']):.3f}",
                refresh=False,
            )
        return evaluation

    evaluation = record_metrics(eval_streams)
    total_training_cases = sum(len(stream.cases) for stream in train_streams)
    timeline_stride = max(1, total_training_cases // 10_000)
    learning_bar = tqdm(
        total=total_training_cases,
        desc="Learning",
        unit="case",
        dynamic_ncols=True,
        disable=not config.progress,
    )
    learning_bar.set_postfix(
        labels=labels_requested,
        agreement=f"{float(metrics_history[-1]['f1']):.3f}",
        refresh=False,
    )
    feedback_path = output_dir / "feedback.jsonl"
    learning_steps_path = output_dir / "learning_steps.csv"
    append_logs = config.resume and feedback_path.exists() and learning_steps_path.exists()
    feedback_file = feedback_path.open(
        "a" if append_logs else "w",
        encoding="utf-8",
    )
    learning_steps_file = learning_steps_path.open(
        "a" if append_logs else "w",
        encoding="utf-8",
        newline="",
    )
    learning_steps_writer = csv.DictWriter(
        learning_steps_file,
        fieldnames=_LEARNING_STEP_FIELDS,
    )
    if not append_logs:
        learning_steps_writer.writeheader()
    try:
        for stream in train_streams:
            for tick, tick_cases in _cases_by_tick(stream.cases):
                features_by_case = [(case, encoder.encode(case)) for case in tick_cases]
                probabilities = [
                    (case, learner.predict_proba(features))
                    for case, features in features_by_case
                ]
                selected_case_ids = query_policy.select_tick(probabilities)
                for case, features in features_by_case:
                    cases_processed += 1
                    learning_bar.update(1)
                    if cases_processed <= skip_cases:
                        continue
                    probability = learner.predict_proba(features)
                    prediction = probability >= config.decision_threshold
                    queried = case.case_id in selected_case_ids
                    feedback: OperatorFeedback | None = None
                    if queried:
                        feedback = _feedback_for_case(case, operator, config.mode)
                        if feedback is not None:
                            learner.update(features, feedback.label)
                            labels_requested += 1
                            learning_bar.set_postfix(
                                labels=labels_requested,
                                agreement=f"{float(metrics_history[-1]['f1']):.3f}",
                                refresh=False,
                            )
                            feedback_file.write(
                                json.dumps(
                                    _feedback_row(
                                        commit_hash,
                                        stream,
                                        case,
                                        probability,
                                        prediction,
                                        feedback,
                                    ),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            if (
                                config.metric_interval > 0
                                and labels_requested % config.metric_interval == 0
                                and labels_requested != last_metric_label_count
                            ):
                                evaluation = record_metrics(metric_eval_streams)
                                last_metric_label_count = labels_requested
                    step_row = _learning_step_row(
                        commit_hash,
                        stream,
                        case,
                        probability,
                        prediction,
                        queried=feedback is not None,
                        feedback=feedback,
                        labels_requested=labels_requested,
                        cases_processed=cases_processed,
                    )
                    learning_steps_writer.writerow(step_row)
                    if cases_processed % timeline_stride == 0:
                        timeline_steps.append(step_row)
    except HumanStop:
        pass
    finally:
        feedback_file.close()
        learning_steps_file.close()
        learning_bar.close()
        learning_bar = None

    evaluation = record_metrics(eval_streams)
    final_metrics = _write_metrics(
        output_dir,
        config,
        train_streams,
        eval_streams,
        operator,
        evaluation,
        metrics_history,
        commit_hash,
    )
    learner.save_json(
        output_dir / "model_weights.json",
        extra={"commit_hash": commit_hash},
    )
    _write_checkpoint(
        checkpoint,
        learner,
        config,
        commit_hash,
        cases_processed=cases_processed,
        labels_requested=labels_requested,
    )
    target_locations = set(evaluation["target_location_ids"])
    predicted_locations = set(evaluation["predicted_location_ids"])
    write_hitl_plots(
        output_dir,
        metrics_history=metrics_history,
        learning_steps=timeline_steps,
        feature_names=encoder.feature_names,
        weights=[float(v) for v in learner.weights],
        commit_hash=commit_hash,
        scenario=(map_stream.scenario if map_stream else None),
        canonical=(map_stream.canonical if map_stream else None),
        predicted_location_ids=predicted_locations,
        target_location_ids=target_locations,
    )
    write_supervisor_artifacts(
        output_dir,
        train_streams=train_streams,
        eval_streams=eval_streams,
        map_stream=map_stream,
        config=config,
        commit_hash=commit_hash,
        predicted_location_ids=predicted_locations,
        target_location_ids=target_locations,
    )
    return HitlRunResult(
        output_dir=output_dir,
        metrics=final_metrics,
        labels_requested=labels_requested,
        cases_processed=cases_processed,
    )


def build_bulgaria_case_streams(
    config: HitlDemoConfig,
) -> tuple[list[CaseStream], list[CaseStream], CaseStream | None]:
    all_streams = build_bulgaria_streams(config)
    train_streams, eval_streams = split_streams_for_evaluation(
        all_streams,
        seeds=config.seeds,
        evaluation_seed=config.evaluation_seed,
        evaluation_fraction=config.evaluation_fraction,
    )
    map_stream = eval_streams[0] if eval_streams else None
    return train_streams, eval_streams, map_stream


def build_bulgaria_streams(config: HitlDemoConfig) -> list[CaseStream]:
    from runtime.experiments import build_bulgaria_simulation_input, run_simulation

    if not config.seeds:
        raise ValueError("at least one seed is required")
    all_streams: list[CaseStream] = []
    total_stages = len(config.seeds) * (2 + len(config.perturbations))
    with tqdm(
        total=total_stages,
        desc="Preparing HITL streams",
        unit="stage",
        dynamic_ncols=True,
        disable=not config.progress,
    ) as preparation_bar:
        for seed in config.seeds:
            preparation_bar.set_postfix(seed=seed, task="scenario", refresh=False)
            sim_input = build_bulgaria_simulation_input(
                geocode=config.geocode,
                geocode_cache_path=config.geocode_cache_path,
                geocode_max_new_requests=config.geocode_max_new_requests,
                seed=seed,
                max_locations=config.max_locations,
                packs_per_obp=config.packs_per_obp,
                packs_per_wholesaler=config.packs_per_wholesaler,
                n_obp_hubs=config.n_obp_hubs,
            )
            preparation_bar.update(1)

            preparation_bar.set_postfix(seed=seed, task="baseline", refresh=False)
            _base_sim, base_records = run_simulation(sim_input.engine_input, config.ticks)
            all_streams.append(
                _case_stream_from_records(
                    stream_id=f"seed{seed}:baseline",
                    records=base_records,
                    inp=sim_input.engine_input,
                    scenario=sim_input.scenario,
                    canonical=sim_input.canonical,
                    config=config,
                    seed=seed,
                )
            )
            preparation_bar.update(1)
            for ix, ptype_name in enumerate(config.perturbations):
                ptype = PerturbationType(ptype_name)
                preparation_bar.set_postfix(seed=seed, task=ptype.value, refresh=False)
                perturbed_input, labels = apply_perturbation(
                    sim_input.engine_input,
                    PerturbationSpec(
                        ptype=ptype,
                        seed=seed * 100 + ix,
                        n_targets=3,
                        intensity=8.0,
                    ),
                )
                _, records = run_simulation(perturbed_input, config.ticks)
                all_streams.append(
                    _case_stream_from_records(
                        stream_id=f"seed{seed}:{ptype.value}",
                        records=records,
                        inp=perturbed_input,
                        scenario=sim_input.scenario,
                        canonical=sim_input.canonical,
                        config=config,
                        seed=seed,
                        perturbation_type=ptype.value,
                        perturbation_labels=labels,
                    )
                )
                preparation_bar.update(1)

    return all_streams


def split_streams_for_evaluation(
    all_streams: Sequence[CaseStream],
    *,
    seeds: Sequence[int],
    evaluation_seed: int | None,
    evaluation_fraction: float,
) -> tuple[list[CaseStream], list[CaseStream]]:
    if len(seeds) == 1:
        split_pairs = [
            _split_stream(stream, evaluation_fraction) for stream in all_streams
        ]
        train_streams = [train for train, _evaluation in split_pairs]
        eval_streams = [evaluation for _train, evaluation in split_pairs]
    else:
        if evaluation_seed is None:
            evaluation_seed = seeds[-1]
        if evaluation_seed not in seeds:
            raise ValueError(
                f"evaluation seed {evaluation_seed} is not present in seeds {tuple(seeds)}"
            )
        train_streams = [
            replace(stream, split="train")
            for stream in all_streams
            if stream.seed != evaluation_seed
        ]
        eval_streams = [
            replace(stream, split="evaluation")
            for stream in all_streams
            if stream.seed == evaluation_seed
        ]
    return train_streams, eval_streams


def evaluate_streams(
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
    encoder: FeatureEncoder,
    learner: OnlineLogisticLearner,
    *,
    threshold: float,
    labels_requested: int,
    cases_processed: int,
) -> dict[str, Any]:
    labels: list[bool] = []
    probabilities: list[float] = []
    by_perturbation: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"labels": [], "probabilities": []})
    by_seed: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"labels": [], "probabilities": []}
    )
    predicted_locations: set[int] = set()
    target_locations: set[int] = set()
    for stream in eval_streams:
        for label in stream.perturbation_labels or []:
            target_locations.add(int(label.location_id))
        stream_labels: list[bool] = []
        stream_probabilities: list[float] = []
        for case in stream.cases:
            prob = learner.predict_proba(encoder.encode(case))
            feedback = operator.label(case)
            probabilities.append(prob)
            labels.append(feedback.label)
            stream_probabilities.append(prob)
            stream_labels.append(feedback.label)
            if prob >= threshold:
                loc = _case_target_location(case)
                if loc is not None:
                    predicted_locations.add(loc)
        key = stream.perturbation_type or "BASELINE"
        by_perturbation[key]["labels"].extend(stream_labels)
        by_perturbation[key]["probabilities"].extend(stream_probabilities)
        seed_key = "unknown" if stream.seed is None else str(stream.seed)
        by_seed[seed_key]["labels"].extend(stream_labels)
        by_seed[seed_key]["probabilities"].extend(stream_probabilities)
    overall = classification_metrics(
        labels,
        probabilities,
        threshold=threshold,
        labels_requested=labels_requested,
        cases_processed=cases_processed,
    ).to_dict()
    per_perturbation = {
        key: classification_metrics(
            [bool(v) for v in rows["labels"]],
            [float(v) for v in rows["probabilities"]],
            threshold=threshold,
            labels_requested=labels_requested,
            cases_processed=cases_processed,
        ).to_dict()
        for key, rows in sorted(by_perturbation.items())
    }
    per_seed = {
        key: classification_metrics(
            [bool(v) for v in rows["labels"]],
            [float(v) for v in rows["probabilities"]],
            threshold=threshold,
            labels_requested=labels_requested,
            cases_processed=cases_processed,
        ).to_dict()
        for key, rows in sorted(by_seed.items())
    }
    return {
        "overall": overall,
        "per_perturbation": per_perturbation,
        "per_seed": per_seed,
        "target_location": target_location_metrics(
            predicted_locations,
            target_locations,
        ).to_dict(),
        "predicted_location_ids": sorted(predicted_locations),
        "target_location_ids": sorted(target_locations),
    }


def repo_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _case_stream_from_records(
    *,
    stream_id: str,
    records: list[dict[str, Any]],
    inp: EngineInput,
    scenario: Scenario,
    canonical: Any,
    config: HitlDemoConfig,
    seed: int | None = None,
    perturbation_type: str | None = None,
    perturbation_labels: list[PerturbationLabel] | None = None,
) -> CaseStream:
    builder = CaseBuilder(
        inp,
        scenario,
        canonical,
        config=CaseBuilderConfig(
            safe_sample_rate=config.safe_sample_rate,
            safe_sample_seed=config.safe_sample_seed,
            recent_window_ticks=config.recent_window_ticks,
            recent_source_candidate_threshold=config.recent_source_candidate_threshold,
        ),
    )
    cases = build_cases_from_records(records, builder, stream_id=stream_id)
    return CaseStream(
        stream_id=stream_id,
        split="all",
        cases=cases,
        records=records,
        engine_input=inp,
        scenario=scenario,
        canonical=canonical,
        perturbation_type=perturbation_type,
        perturbation_labels=perturbation_labels or [],
        seed=seed,
    )


def _split_stream(stream: CaseStream, evaluation_fraction: float) -> tuple[CaseStream, CaseStream]:
    train_cases: list[AlertCase] = []
    eval_cases: list[AlertCase] = []
    for case in stream.cases:
        if _hash_unit("eval", case.case_id) < evaluation_fraction:
            eval_cases.append(case)
        else:
            train_cases.append(case)
    common = {
        "records": stream.records,
        "engine_input": stream.engine_input,
        "scenario": stream.scenario,
        "canonical": stream.canonical,
        "perturbation_type": stream.perturbation_type,
        "perturbation_labels": stream.perturbation_labels,
        "seed": stream.seed,
    }
    return (
        CaseStream(stream_id=stream.stream_id, split="train", cases=train_cases, **common),
        CaseStream(stream_id=stream.stream_id, split="evaluation", cases=eval_cases, **common),
    )


def _cases_by_tick(cases: Sequence[AlertCase]) -> list[tuple[int, list[AlertCase]]]:
    grouped: dict[int, list[AlertCase]] = defaultdict(list)
    for case in cases:
        grouped[int(case.tick)].append(case)
    return [(tick, grouped[tick]) for tick in sorted(grouped)]


def _sample_evaluation_streams(
    streams: Sequence[CaseStream],
    *,
    max_cases: int,
) -> list[CaseStream]:
    total = sum(len(stream.cases) for stream in streams)
    if max_cases <= 0 or total <= max_cases:
        return list(streams)
    selected = heapq.nsmallest(
        max_cases,
        (
            (_hash_unit("metric-eval", case.case_id), case.case_id)
            for stream in streams
            for case in stream.cases
        ),
    )
    selected_ids = {case_id for _score, case_id in selected}
    return [
        replace(
            stream,
            cases=[case for case in stream.cases if case.case_id in selected_ids],
        )
        for stream in streams
    ]


def _feedback_for_case(
    case: AlertCase,
    operator: HiddenOperator,
    mode: str,
) -> OperatorFeedback | None:
    if mode == "oracle":
        return operator.label(case)
    if mode == "human":
        return _prompt_human(case)
    raise ValueError(f"unknown HITL mode {mode!r}")


def _prompt_human(case: AlertCase) -> OperatorFeedback | None:
    print(
        json.dumps(
            {
                "case_id": case.case_id,
                "tick": case.tick,
                "event_type": case.event_type,
                "alert_code": case.alert_code,
                "decommission_reason": case.decommission_reason,
                "from_role": case.from_role,
                "to_role": case.to_role,
                "from_market": case.from_market,
                "to_market": case.to_market,
            },
            indent=2,
        )
    )
    while True:
        choice = input("[u]nsafe [s]afe s[k]ip [q]uit: ").strip().lower()
        if choice == "u":
            return OperatorFeedback(label=True, confidence=1.0, rule_id="human")
        if choice == "s":
            return OperatorFeedback(label=False, confidence=1.0, rule_id="human")
        if choice == "k":
            return None
        if choice == "q":
            raise HumanStop


def _validate_class_counts(
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
) -> None:
    labels = [operator.label(case).label for stream in (*train_streams, *eval_streams) for case in stream.cases]
    if not labels:
        raise RuntimeError("HITL case stream is empty")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise RuntimeError(
            f"HITL case stream must contain safe and unsafe cases; got unsafe={positives}, safe={negatives}"
        )


def _write_config(
    output_dir: Path,
    config: HitlDemoConfig,
    commit_hash: str,
    feature_names: Sequence[str],
    *,
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
) -> None:
    payload = {
        "commit_hash": commit_hash,
        "config": _config_payload(config),
        "feature_count": len(feature_names),
        "resolved_training_seeds": _stream_seeds(train_streams),
        "resolved_evaluation_seeds": _stream_seeds(eval_streams),
        "spec_decisions": [
            "same_city uses canonical city when available, otherwise equal postal code",
            "evaluation labels are produced after prediction and never update the learner",
            "multi-seed runs hold out one complete seed; single-seed smoke runs use a case split",
            "intermediate curves use a fixed bounded evaluation sample; final metrics use the full holdout",
            "oracle replay is implemented; human mode is terminal/checkpoint-capable",
        ],
    }
    (output_dir / "config_resolved.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_cases(
    output_dir: Path,
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    commit_hash: str,
    *,
    progress: bool,
) -> None:
    streams = (*train_streams, *eval_streams)
    total = sum(len(stream.cases) for stream in streams)
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as f:
        with tqdm(
            total=total,
            desc="Writing cases",
            unit="case",
            dynamic_ncols=True,
            disable=not progress,
        ) as writing_bar:
            for stream in streams:
                for case in stream.cases:
                    f.write(
                        json.dumps(
                            {
                                "commit_hash": commit_hash,
                                "split": stream.split,
                                "stream_id": stream.stream_id,
                                "seed": stream.seed,
                                "perturbation_type": stream.perturbation_type,
                                "case": to_jsonable(case),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    writing_bar.update(1)


def _write_feature_names(output_dir: Path, feature_names: Sequence[str], commit_hash: str) -> None:
    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {"commit_hash": commit_hash, "feature_names": list(feature_names)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _feedback_row(
    commit_hash: str,
    stream: CaseStream,
    case: AlertCase,
    probability: float,
    prediction: bool,
    feedback: OperatorFeedback,
) -> dict[str, Any]:
    return {
        "commit_hash": commit_hash,
        "stream_id": stream.stream_id,
        "case_id": case.case_id,
        "tick": case.tick,
        "probability": probability,
        "prediction": prediction,
        "operator_label": feedback.label,
        "operator_confidence": feedback.confidence,
        "operator_rule_id": feedback.rule_id,
    }


def _learning_step_row(
    commit_hash: str,
    stream: CaseStream,
    case: AlertCase,
    probability: float,
    prediction: bool,
    *,
    queried: bool,
    feedback: OperatorFeedback | None,
    labels_requested: int,
    cases_processed: int,
) -> dict[str, Any]:
    return {
        "commit_hash": commit_hash,
        "stream_id": stream.stream_id,
        "perturbation_type": stream.perturbation_type or "",
        "case_id": case.case_id,
        "tick": case.tick,
        "event_type": case.event_type,
        "alert_code": case.alert_code or "",
        "probability": f"{probability:.12g}",
        "prediction": prediction,
        "queried": queried,
        "operator_label": "" if feedback is None else feedback.label,
        "operator_confidence": "" if feedback is None else f"{feedback.confidence:.6g}",
        "operator_rule_id": "" if feedback is None else feedback.rule_id,
        "labels_requested": labels_requested,
        "cases_processed": cases_processed,
    }


def _write_metrics(
    output_dir: Path,
    config: HitlDemoConfig,
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
    evaluation: dict[str, Any],
    metrics_history: Sequence[Mapping[str, Any]],
    commit_hash: str,
) -> dict[str, Any]:
    class_counts = _class_counts(train_streams, eval_streams, operator)
    initial_f1 = float(metrics_history[0]["f1"]) if metrics_history else 0.0
    final_f1 = float(evaluation["overall"]["f1"])
    payload = {
        "commit_hash": commit_hash,
        "config": _config_payload(config),
        "class_counts": class_counts,
        "resolved_training_seeds": _stream_seeds(train_streams),
        "resolved_evaluation_seeds": _stream_seeds(eval_streams),
        "metrics_history": list(metrics_history),
        "final": evaluation["overall"],
        "per_perturbation": evaluation["per_perturbation"],
        "per_seed": evaluation["per_seed"],
        "target_location": evaluation["target_location"],
        "acceptance_status": {
            "case_stream_has_safe_and_unsafe": class_counts["safe"] > 0 and class_counts["unsafe"] > 0,
            "final_f1_at_least_0_85": final_f1 >= 0.85,
            "final_fpr_at_most_0_10": float(evaluation["overall"]["false_positive_rate"]) <= 0.10,
            "f1_improved_by_0_20": (final_f1 - initial_f1) >= 0.20,
        },
        "acceptance_not_checked_in_single_policy_run": [
            "at least four of five seed holdouts meet the F1 target",
            "uncertainty mode uses at most half the labels",
            "uncertainty F1 stays within 0.05 of query-all",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _write_checkpoint(
    path: Path,
    learner: OnlineLogisticLearner,
    config: HitlDemoConfig,
    commit_hash: str,
    *,
    cases_processed: int,
    labels_requested: int,
) -> None:
    payload = {
        "commit_hash": commit_hash,
        "config": _config_payload(config),
        "learner": learner.to_dict(),
        "cases_processed": cases_processed,
        "labels_requested": labels_requested,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _class_counts(
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
) -> dict[str, int]:
    labels = [operator.label(case).label for stream in (*train_streams, *eval_streams) for case in stream.cases]
    unsafe = sum(labels)
    return {"unsafe": unsafe, "safe": len(labels) - unsafe, "total": len(labels)}


def _config_payload(config: HitlDemoConfig) -> dict[str, Any]:
    payload = asdict(config)
    return {k: str(v) if isinstance(v, Path) else v for k, v in payload.items()}


def _stream_seeds(streams: Sequence[CaseStream]) -> list[int]:
    return sorted({int(stream.seed) for stream in streams if stream.seed is not None})


def _case_target_location(case: AlertCase) -> int | None:
    if case.to_location_id is not None:
        return int(case.to_location_id)
    if case.from_location_id is not None:
        return int(case.from_location_id)
    return None


def _hash_unit(*parts: str) -> float:
    import hashlib

    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)
