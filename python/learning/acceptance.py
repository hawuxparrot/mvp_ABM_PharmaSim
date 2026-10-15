from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from tqdm.auto import tqdm

from .features import FeatureEncoder
from .online_model import LearnerConfig, OnlineLogisticLearner
from .operator import HiddenOperator
from .query_policy import QueryPolicyConfig, make_query_policy
from .runner import CaseStream, HitlDemoConfig, _cases_by_tick, evaluate_streams


def run_seed_holdout_acceptance(
    streams: Sequence[CaseStream],
    operator: HiddenOperator,
    *,
    config: HitlDemoConfig,
) -> dict[str, Any]:
    seeds = sorted({stream.seed for stream in streams if stream.seed is not None})
    if len(seeds) < 5:
        raise ValueError("acceptance requires at least five simulation seeds")

    work_per_policy = sum(
        len(stream.cases)
        for held_seed in seeds
        for stream in streams
        if stream.seed != held_seed
    )
    policy_results: dict[str, Any] = {}
    with tqdm(
        total=work_per_policy * 2,
        desc="Seed-holdout acceptance",
        unit="case",
        dynamic_ncols=True,
        disable=not config.progress,
    ) as progress_bar:
        for policy_name in ("all", "uncertainty"):
            folds: dict[str, Any] = {}
            for held_seed in seeds:
                progress_bar.set_postfix(
                    policy=policy_name,
                    held_seed=held_seed,
                    refresh=False,
                )
                train_streams = [
                    replace(stream, split="train")
                    for stream in streams
                    if stream.seed != held_seed
                ]
                eval_streams = [
                    replace(stream, split="evaluation")
                    for stream in streams
                    if stream.seed == held_seed
                ]
                folds[str(held_seed)] = _run_holdout_fold(
                    train_streams,
                    eval_streams,
                    operator,
                    config=replace(config, query_policy=policy_name),
                    progress_bar=progress_bar,
                )
            policy_results[policy_name] = {
                "per_seed": folds,
                "initial": _aggregate_metrics(
                    [fold["initial"] for fold in folds.values()]
                ),
                "final": _aggregate_metrics(
                    [fold["final"] for fold in folds.values()]
                ),
                "labels_requested": sum(
                    int(fold["labels_requested"]) for fold in folds.values()
                ),
                "cases_processed": sum(
                    int(fold["cases_processed"]) for fold in folds.values()
                ),
            }

    query_all = policy_results["all"]
    uncertainty = policy_results["uncertainty"]
    query_all_seed_passes = sum(
        float(fold["final"]["f1"]) >= 0.85
        for fold in query_all["per_seed"].values()
    )
    status = {
        "case_stream_has_safe_and_unsafe": (
            int(query_all["final"]["true_positives"])
            + int(query_all["final"]["false_negatives"])
            > 0
            and int(query_all["final"]["true_negatives"])
            + int(query_all["final"]["false_positives"])
            > 0
        ),
        "query_all_final_f1_at_least_0_85": float(query_all["final"]["f1"]) >= 0.85,
        "query_all_final_fpr_at_most_0_10": (
            float(query_all["final"]["false_positive_rate"]) <= 0.10
        ),
        "query_all_f1_improved_by_0_20": (
            float(query_all["final"]["f1"]) - float(query_all["initial"]["f1"])
            >= 0.20
        ),
        "at_least_four_of_five_seed_holdouts_meet_f1": query_all_seed_passes >= 4,
        "uncertainty_uses_at_most_half_the_labels": (
            int(uncertainty["labels_requested"])
            <= 0.5 * int(query_all["labels_requested"])
        ),
        "uncertainty_f1_within_0_05_of_query_all": (
            float(uncertainty["final"]["f1"])
            >= float(query_all["final"]["f1"]) - 0.05
        ),
    }
    return {
        "seeds": seeds,
        "policies": policy_results,
        "query_all_seed_passes": query_all_seed_passes,
        "acceptance_status": status,
        "passed": all(status.values()),
    }


def _run_holdout_fold(
    train_streams: Sequence[CaseStream],
    eval_streams: Sequence[CaseStream],
    operator: HiddenOperator,
    *,
    config: HitlDemoConfig,
    progress_bar: tqdm,
) -> dict[str, Any]:
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
    query_policy = make_query_policy(
        config.query_policy,
        QueryPolicyConfig(
            query_budget_per_tick=config.query_budget_per_tick,
            uncertainty_margin=config.uncertainty_margin,
            exploration_rate=config.exploration_rate,
            seed=config.safe_sample_seed + 101,
        ),
    )
    cases_processed = 0
    labels_requested = 0
    initial = evaluate_streams(
        eval_streams,
        operator,
        encoder,
        learner,
        threshold=config.decision_threshold,
        labels_requested=0,
        cases_processed=0,
    )["overall"]
    for stream in train_streams:
        for _tick, tick_cases in _cases_by_tick(stream.cases):
            encoded = [(case, encoder.encode(case)) for case in tick_cases]
            probabilities = [
                (case, learner.predict_proba(features))
                for case, features in encoded
            ]
            selected_ids = query_policy.select_tick(probabilities)
            for case, features in encoded:
                cases_processed += 1
                if case.case_id in selected_ids:
                    learner.update(features, operator.label(case).label)
                    labels_requested += 1
                progress_bar.update(1)
    final = evaluate_streams(
        eval_streams,
        operator,
        encoder,
        learner,
        threshold=config.decision_threshold,
        labels_requested=labels_requested,
        cases_processed=cases_processed,
    )["overall"]
    return {
        "initial": initial,
        "final": final,
        "labels_requested": labels_requested,
        "cases_processed": cases_processed,
    }


def _aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    tp = sum(int(row["true_positives"]) for row in metrics)
    fp = sum(int(row["false_positives"]) for row in metrics)
    tn = sum(int(row["true_negatives"]) for row in metrics)
    fn = sum(int(row["false_negatives"]) for row in metrics)
    total = tp + fp + tn + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    weighted_log_loss = _safe_div(
        sum(
            float(row["log_loss"])
            * (
                int(row["true_positives"])
                + int(row["false_positives"])
                + int(row["true_negatives"])
                + int(row["false_negatives"])
            )
            for row in metrics
        ),
        total,
    )
    probability_mae = _safe_div(
        sum(
            float(row["probability_mae"])
            * (
                int(row["true_positives"])
                + int(row["false_positives"])
                + int(row["true_negatives"])
                + int(row["false_negatives"])
            )
            for row in metrics
        ),
        total,
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "oracle_unsafe_rate": _safe_div(tp + fn, total),
        "learned_unsafe_rate": _safe_div(tp + fp, total),
        "disagreement_rate": _safe_div(fp + fn, total),
        "probability_mae": probability_mae,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2.0 * precision * recall, precision + recall),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "log_loss": weighted_log_loss,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
