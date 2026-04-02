#pragma once

#include "engine_input.hpp"

#include <cstdint>
#include <vector>

/// Mutable per-pack state for the simulation kernel (SoA). Initialized from ``EngineInput`` pack columns.
struct SimulationState {
    std::vector<std::uint8_t> pack_state;
    std::vector<std::uint32_t> pack_location_id;
    std::vector<std::uint32_t> pack_market_id;

    SimulationState() = default;

    explicit SimulationState(const EngineInput& in);
};
