#pragma once

#include <cstdint>

enum class ORG_TYPE: std::uint8_t {
    OBP = 0,
    WHOLESALER = 1,
    LOCAL_ORG = 2,
    NMVO = 3,
    EMVO = 4,
};

enum class PACK_STATE: std::uint8_t {
    UPLOADED = 0,
    ACTIVE = 1,
    DECOMISSIONED = 2,
};

enum LOCATION_BEHAVIOR {
    HAS_BEHAVIOR = 1,
    NO_BEHAVIOR = 0,
};