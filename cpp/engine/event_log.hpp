#pragma once

#include "enums.hpp"

#include <cstdint>
#include <limits>
#include <vector>

/// Sentinel for ``from`` / ``to`` when unused (VERIFY, DECOMMISSION, REACTIVATE).
inline constexpr std::uint32_t k_event_no_location = std::numeric_limits<std::uint32_t>::max();
/// Sentinel for ``event_market_id`` when an event is not scoped to a national NMVS.
inline constexpr std::uint32_t k_event_no_market = std::numeric_limits<std::uint32_t>::max();
/// ``event_detail`` value when the event type carries no detail code.
inline constexpr std::uint8_t k_event_no_detail = 0u;

/// Append-only event trail (one row per event).
///
/// ``event_detail`` overloads on ``event_type``: for DECOMMISSION it holds the
/// ``DECOMMISSION_REASON``, for ALERT it holds the ``ALERT_CODE``, otherwise it is unused.
/// ``event_market_id`` is the national NMVS the transaction hit (the scan location's market).
struct EventLog {
    std::vector<std::uint64_t> tick;
    std::vector<std::uint32_t> pack_id;
    std::vector<std::uint8_t> event_type;
    std::vector<std::uint32_t> from_location_id;
    std::vector<std::uint32_t> to_location_id;
    std::vector<std::uint8_t> event_detail;
    std::vector<std::uint32_t> event_market_id;

    void push(
        std::uint64_t t,
        std::uint32_t pid,
        EventType type,
        std::uint32_t from_loc = k_event_no_location,
        std::uint32_t to_loc = k_event_no_location,
        std::uint8_t detail = k_event_no_detail,
        std::uint32_t market = k_event_no_market
    ) {
        tick.push_back(t);
        pack_id.push_back(pid);
        event_type.push_back(static_cast<std::uint8_t>(type));
        from_location_id.push_back(from_loc);
        to_location_id.push_back(to_loc);
        event_detail.push_back(detail);
        event_market_id.push_back(market);
    }
};
