#pragma once

#include "enums.hpp"

#include <cstdint>
#include <limits>
#include <vector>

/// Sentinel for ``from`` / ``to`` when unused (VERIFY, DECOMMISSION, REACTIVATE).
inline constexpr std::uint32_t k_event_no_location = std::numeric_limits<std::uint32_t>::max();

/// Append-only event trail (one row per event).
struct EventLog {
    std::vector<std::uint64_t> tick;
    std::vector<std::uint32_t> pack_id;
    std::vector<std::uint8_t> event_type;
    std::vector<std::uint32_t> from_location_id;
    std::vector<std::uint32_t> to_location_id;

    void push(
        std::uint64_t t,
        std::uint32_t pid,
        EventType type,
        std::uint32_t from_loc = k_event_no_location,
        std::uint32_t to_loc = k_event_no_location
    ) {
        tick.push_back(t);
        pack_id.push_back(pid);
        event_type.push_back(static_cast<std::uint8_t>(type));
        from_location_id.push_back(from_loc);
        to_location_id.push_back(to_loc);
    }
};
