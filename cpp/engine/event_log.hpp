#pragma once

#include <cstdint>
#include <vector>

/// Append-only event trail (extend with event_type, prev/new state, etc. when FSM lands).
struct EventLog {
    std::vector<std::uint64_t> tick;
    std::vector<std::uint32_t> pack_id;
};
