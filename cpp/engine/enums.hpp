#pragma once

#include <cstdint>

/// Must match ``compiler.types.ENGINE_INPUT_SCHEMA_VERSION`` in Python (defined in compiler/types.py)..
constexpr const char* ENGINE_INPUT_SCHEMA_VERSION = "engine_input.v8";

enum class ORG_TYPE : std::uint8_t {
    OBP = 0,
    WHOLESALER = 1,
    LOCAL_ORG = 2,
    NMVO = 3,
    EMVO = 4,
};

enum class PACK_STATE : std::uint8_t {
    UPLOADED = 0,
    ACTIVE = 1,
    DECOMISSIONED = 2,
};

/// EMVS decommission status reasons. Must match ``compiler.enums.DECOMMISSION_REASON_TO_U8``.
enum class DECOMMISSION_REASON : std::uint8_t {
    DISPENSED = 0,
    STOLEN = 1,
    EXPORTED = 2,
    DESTROYED = 3,
    SAMPLED = 4,
    LOCKED = 5,
    RECALLED = 6,
};

/// EMVS verification alert codes. Must match ``compiler.enums.ALERT_CODE_TO_U8``.
/// Carried in ``EventLog.event_detail`` when ``event_type == ALERT``.
enum class ALERT_CODE : std::uint8_t {
    UNKNOWN_SERIAL = 0,
    OUT_OF_MARKET = 1,
    ALREADY_DECOMMISSIONED = 2,
    DUPLICATE_SERIAL = 3,
};

enum class LOCATION_BEHAVIOR : std::uint8_t {
    NO_BEHAVIOR = 0,
    HAS_BEHAVIOR = 1,
};

enum class EventType : std::uint8_t {
    VERIFY = 0,
    DECOMMISSION = 1,
    REACTIVATE = 2,
    MOVE = 3,
    /// Placeholder for future ground-truth vs registry mismatch alerts.
    REGISTRY_SYNC = 4,
    /// EMVS verification alert; ``event_detail`` carries the ``ALERT_CODE``.
    ALERT = 5,
};
