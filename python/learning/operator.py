from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AlertCase, OperatorFeedback


@dataclass(frozen=True)
class OperatorRule:
    rule_id: str
    label: bool
    confidence: float
    when: dict[str, Any]


class HiddenOperator:
    def __init__(
        self,
        rules: list[OperatorRule],
        *,
        thresholds: dict[str, float],
        default_label: bool = False,
        default_confidence: float = 0.60,
        default_rule_id: str = "default_safe",
    ):
        self.rules = list(rules)
        self.thresholds = dict(thresholds)
        self.default_label = bool(default_label)
        self.default_confidence = float(default_confidence)
        self.default_rule_id = default_rule_id

    @classmethod
    def from_json(cls, path: str | Path) -> "HiddenOperator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rules = [
            OperatorRule(
                rule_id=str(row["id"]),
                label=bool(row["label"]),
                confidence=float(row.get("confidence", 1.0)),
                when=dict(row.get("when", {})),
            )
            for row in payload.get("rules", [])
        ]
        default = payload.get("default", {})
        return cls(
            rules,
            thresholds={k: float(v) for k, v in payload.get("thresholds", {}).items()},
            default_label=bool(default.get("label", False)),
            default_confidence=float(default.get("confidence", 0.60)),
            default_rule_id=str(default.get("rule_id", "default_safe")),
        )

    def label(self, case: AlertCase) -> OperatorFeedback:
        for rule in self.rules:
            if _matches(case, rule.when, self.thresholds):
                return OperatorFeedback(
                    label=rule.label,
                    confidence=rule.confidence,
                    rule_id=rule.rule_id,
                )
        return OperatorFeedback(
            label=self.default_label,
            confidence=self.default_confidence,
            rule_id=self.default_rule_id,
        )


def _matches(case: AlertCase, condition: dict[str, Any], thresholds: dict[str, float]) -> bool:
    if "all" in condition:
        return all(_matches(case, dict(c), thresholds) for c in condition["all"])
    if "any" in condition:
        return any(_matches(case, dict(c), thresholds) for c in condition["any"])
    if "not" in condition:
        return not _matches(case, dict(condition["not"]), thresholds)
    for field_name, expected in condition.items():
        actual = getattr(case, field_name)
        if not _field_matches(actual, expected, thresholds):
            return False
    return True


def _field_matches(actual: Any, expected: Any, thresholds: dict[str, float]) -> bool:
    if isinstance(expected, dict):
        for op, raw_value in expected.items():
            value = _resolve_value(raw_value, thresholds)
            if op == "in":
                if actual not in value:
                    return False
            elif op == "lt":
                if actual is None or not float(actual) < float(value):
                    return False
            elif op == "lte":
                if actual is None or not float(actual) <= float(value):
                    return False
            elif op == "gt":
                if actual is None or not float(actual) > float(value):
                    return False
            elif op == "gte":
                if actual is None or not float(actual) >= float(value):
                    return False
            elif op == "is_none":
                if (actual is None) != bool(value):
                    return False
            else:
                raise ValueError(f"unknown operator rule comparison {op!r}")
        return True
    return actual == _resolve_value(expected, thresholds)


def _resolve_value(raw: Any, thresholds: dict[str, float]) -> Any:
    if isinstance(raw, str) and raw.startswith("threshold."):
        key = raw.split(".", 1)[1]
        if key not in thresholds:
            raise KeyError(f"missing operator threshold {key!r}")
        return thresholds[key]
    if isinstance(raw, list):
        return [_resolve_value(v, thresholds) for v in raw]
    return raw
