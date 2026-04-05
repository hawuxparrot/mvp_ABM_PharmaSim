#pragma once

#include "engine_input.hpp"

#include <cstdint>
#include <vector>

/// Mutable per-pack state for the simulation kernel (SoA). Initialized from ``EngineInput`` pack columns.
struct SimulationState {
    /// Physical ground truth (authoritative for the MVP kernel).
    std::vector<std::uint8_t> pack_state;
    std::vector<std::uint32_t> pack_location_id;
    std::vector<std::uint32_t> pack_market_id;

    /// NMVO/EMVO registry mirror (v1: identical copy each step; later decoupled).
    std::vector<std::uint8_t> registry_pack_state;
    std::vector<std::uint32_t> registry_pack_location_id;
    std::vector<std::uint32_t> registry_pack_market_id;

    /// ``org_type`` of the organization owning each location (by location id).
    std::vector<std::uint8_t> location_org_type;

    SimulationState() = default;

    explicit SimulationState(const EngineInput& in);

    void sync_registry_from_physical(std::uint32_t pack_index);
};
