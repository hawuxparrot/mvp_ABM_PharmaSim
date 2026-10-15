from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .models import AlertCase


@dataclass(frozen=True)
class QueryPolicyConfig:
    query_budget_per_tick: int | None = None
    uncertainty_margin: float = 0.15
    exploration_rate: float = 0.05
    seed: int = 811


class QueryPolicy:
    def select_tick(self, cases_and_probabilities: list[tuple[AlertCase, float]]) -> set[str]:
        raise NotImplementedError


class QueryAllPolicy(QueryPolicy):
    def select_tick(self, cases_and_probabilities: list[tuple[AlertCase, float]]) -> set[str]:
        return {case.case_id for case, _prob in cases_and_probabilities}


class UncertaintyQueryPolicy(QueryPolicy):
    def __init__(self, config: QueryPolicyConfig | None = None):
        self.config = config or QueryPolicyConfig()

    def select_tick(self, cases_and_probabilities: list[tuple[AlertCase, float]]) -> set[str]:
        candidates: list[tuple[float, float, str]] = []
        for case, probability in cases_and_probabilities:
            distance = abs(float(probability) - 0.5)
            explore = _hash_unit(str(self.config.seed), case.case_id) < self.config.exploration_rate
            uncertain = distance <= self.config.uncertainty_margin
            if uncertain or explore:
                candidates.append((distance, _hash_unit("tie", case.case_id), case.case_id))
        candidates.sort()
        if self.config.query_budget_per_tick is not None:
            candidates = candidates[: max(0, int(self.config.query_budget_per_tick))]
        return {case_id for _distance, _tie, case_id in candidates}


def make_query_policy(name: str, config: QueryPolicyConfig | None = None) -> QueryPolicy:
    if name == "all":
        return QueryAllPolicy()
    if name in {"uncertainty", "uncertainty-query"}:
        return UncertaintyQueryPolicy(config)
    raise ValueError(f"unknown query policy {name!r}")


def _hash_unit(*parts: str) -> float:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)
