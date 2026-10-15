from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LearnerConfig:
    learning_rate: float = 0.05
    l2_strength: float = 0.0001
    decision_threshold: float = 0.5
    initial_bias: float = -1.0


class OnlineLogisticLearner:
    def __init__(
        self,
        feature_names: list[str],
        *,
        config: LearnerConfig | None = None,
        weights: np.ndarray | None = None,
        updates: int = 0,
    ):
        self.feature_names = list(feature_names)
        self.config = config or LearnerConfig()
        if weights is None:
            self.weights = np.zeros(len(self.feature_names), dtype=np.float64)
            self.weights[0] = self.config.initial_bias
        else:
            self.weights = np.asarray(weights, dtype=np.float64).copy()
        if self.weights.shape != (len(self.feature_names),):
            raise ValueError("weights length must match feature_names")
        self.updates = int(updates)

    def predict_proba(self, features: np.ndarray) -> float:
        logit = float(np.dot(self.weights, features))
        return _sigmoid(logit)

    def predict(self, features: np.ndarray) -> bool:
        return self.predict_proba(features) >= self.config.decision_threshold

    def loss(self, features: np.ndarray, label: bool) -> float:
        y = 1.0 if label else 0.0
        p = min(1.0 - 1e-12, max(1e-12, self.predict_proba(features)))
        data_loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * self.config.l2_strength * float(np.dot(self.weights[1:], self.weights[1:]))
        return float(data_loss + reg)

    def update(self, features: np.ndarray, label: bool) -> float:
        y = 1.0 if label else 0.0
        p = self.predict_proba(features)
        gradient = (p - y) * features
        gradient[1:] += self.config.l2_strength * self.weights[1:]
        self.weights -= self.config.learning_rate * gradient
        self.updates += 1
        return self.loss(features, label)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "weights": [float(v) for v in self.weights],
            "updates": self.updates,
            "config": {
                "learning_rate": self.config.learning_rate,
                "l2_strength": self.config.l2_strength,
                "decision_threshold": self.config.decision_threshold,
                "initial_bias": self.config.initial_bias,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OnlineLogisticLearner":
        config_payload = dict(payload.get("config", {}))  # type: ignore[arg-type]
        config = LearnerConfig(
            learning_rate=float(config_payload.get("learning_rate", 0.05)),
            l2_strength=float(config_payload.get("l2_strength", 0.0001)),
            decision_threshold=float(config_payload.get("decision_threshold", 0.5)),
            initial_bias=float(config_payload.get("initial_bias", -1.0)),
        )
        return cls(
            [str(v) for v in payload["feature_names"]],  # type: ignore[index]
            config=config,
            weights=np.asarray(payload["weights"], dtype=np.float64),  # type: ignore[index]
            updates=int(payload.get("updates", 0)),
        )

    def save_json(self, path: str | Path, *, extra: dict[str, object] | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "OnlineLogisticLearner":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _sigmoid(logit: float) -> float:
    x = min(40.0, max(-40.0, logit))
    return float(1.0 / (1.0 + np.exp(-x)))
