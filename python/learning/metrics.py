from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    oracle_unsafe_rate: float
    learned_unsafe_rate: float
    disagreement_rate: float
    probability_mae: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    log_loss: float
    labels_requested: int
    cases_processed: int
    query_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "oracle_unsafe_rate": self.oracle_unsafe_rate,
            "learned_unsafe_rate": self.learned_unsafe_rate,
            "disagreement_rate": self.disagreement_rate,
            "probability_mae": self.probability_mae,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "log_loss": self.log_loss,
            "labels_requested": self.labels_requested,
            "cases_processed": self.cases_processed,
            "query_rate": self.query_rate,
        }


@dataclass(frozen=True)
class TargetLocationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
        }


def classification_metrics(
    labels: list[bool],
    probabilities: list[float],
    *,
    threshold: float,
    labels_requested: int,
    cases_processed: int,
) -> ClassificationMetrics:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal length")
    if not labels:
        return ClassificationMetrics(
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            oracle_unsafe_rate=0.0,
            learned_unsafe_rate=0.0,
            disagreement_rate=0.0,
            probability_mae=0.0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            false_positive_rate=0.0,
            log_loss=0.0,
            labels_requested=labels_requested,
            cases_processed=cases_processed,
            query_rate=0.0,
        )
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=np.float64)
    pred = p >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    loss = -np.mean(y * np.log(clipped) + (~y) * np.log(1.0 - clipped))
    return ClassificationMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        oracle_unsafe_rate=float(np.mean(y)),
        learned_unsafe_rate=float(np.mean(pred)),
        disagreement_rate=float(np.mean(pred != y)),
        probability_mae=float(np.mean(np.abs(p - y.astype(np.float64)))),
        accuracy=_safe_div(tp + tn, len(y)),
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=_safe_div(fp, fp + tn),
        log_loss=float(loss),
        labels_requested=labels_requested,
        cases_processed=cases_processed,
        query_rate=_safe_div(labels_requested, cases_processed),
    )


def target_location_metrics(
    predicted_location_ids: set[int],
    target_location_ids: set[int],
) -> TargetLocationMetrics:
    tp = len(predicted_location_ids & target_location_ids)
    fp = len(predicted_location_ids - target_location_ids)
    fn = len(target_location_ids - predicted_location_ids)
    return TargetLocationMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=_safe_div(tp, tp + fp),
        recall=_safe_div(tp, tp + fn),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
