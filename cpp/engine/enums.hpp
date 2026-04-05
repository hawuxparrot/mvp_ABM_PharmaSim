#pragma once

#include <cstdint>

/// Must match ``compiler.types.ENGINE_INPUT_SCHEMA_VERSION`` in Python.
constexpr const char* ENGINE_INPUT_SCHEMA_VERSION = "engine_input.v3";

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
};
