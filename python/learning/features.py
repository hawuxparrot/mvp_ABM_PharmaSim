from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AlertCase


EVENT_TYPES = ("VERIFY", "DECOMMISSION", "REACTIVATE", "MOVE", "REGISTRY_SYNC", "ALERT")
ALERT_CODES = (
    "UNKNOWN_SERIAL",
    "OUT_OF_MARKET",
    "ALREADY_DECOMMISSIONED",
    "DUPLICATE_SERIAL",
    "__NONE__",
    "__OTHER__",
)
ROLES = (
    "OBP",
    "WHOLESALER",
    "LOCAL_ORG",
    "PHARMACY",
    "HOSPITAL",
    "NMVO",
    "EMVO",
    "UNKNOWN",
)


@dataclass(frozen=True)
class FeatureConfig:
    distance_scale_km: float = 100.0
    pack_event_scale: float = 25.0
    pack_alert_scale: float = 5.0
    recent_event_scale: float = 25.0


class FeatureEncoder:
    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.feature_names = self._feature_names()

    def encode(self, case: AlertCase) -> np.ndarray:
        vals: list[float] = [1.0]
        vals.extend(_one_hot(case.event_type, EVENT_TYPES))
        vals.extend(_one_hot(_alert_code_value(case.alert_code), ALERT_CODES))
        vals.extend(_one_hot(_role_value(case.from_role), ROLES))
        vals.extend(_one_hot(_role_value(case.to_role), ROLES))
        vals.extend(
            [
                float(case.same_location),
                float(case.same_organization),
                float(case.same_city),
                float(case.same_postal_code),
                _norm_optional(case.distance_km, self.config.distance_scale_km),
                float(case.market_mismatch),
                _norm_count(case.prior_pack_events, self.config.pack_event_scale),
                _norm_count(case.prior_pack_alerts, self.config.pack_alert_scale),
                _norm_count(case.source_recent_event_count, self.config.recent_event_scale),
                _norm_count(case.destination_recent_event_count, self.config.recent_event_scale),
                _clip01(case.source_trust_score),
                _clip01(case.destination_trust_score),
            ]
        )
        return np.asarray(vals, dtype=np.float64)

    def _feature_names(self) -> list[str]:
        names = ["bias"]
        names.extend(f"event_type={v}" for v in EVENT_TYPES)
        names.extend(f"alert_code={v}" for v in ALERT_CODES)
        names.extend(f"from_role={v}" for v in ROLES)
        names.extend(f"to_role={v}" for v in ROLES)
        names.extend(
            [
                "same_location",
                "same_organization",
                "same_city",
                "same_postal_code",
                "distance_km_normalized",
                "market_mismatch",
                "prior_pack_events_normalized",
                "prior_pack_alerts_normalized",
                "source_recent_event_count_normalized",
                "destination_recent_event_count_normalized",
                "source_trust_score_normalized",
                "destination_trust_score_normalized",
            ]
        )
        return names


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def _alert_code_value(value: str | None) -> str:
    if value is None:
        return "__NONE__"
    return value if value in ALERT_CODES else "__OTHER__"


def _role_value(value: str | None) -> str:
    if value is None:
        return "UNKNOWN"
    return value if value in ROLES else "UNKNOWN"


def _norm_optional(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    return min(10.0, max(0.0, float(value) / max(scale, 1e-9)))


def _norm_count(value: int, scale: float) -> float:
    return min(10.0, max(0.0, float(value) / max(scale, 1e-9)))


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
