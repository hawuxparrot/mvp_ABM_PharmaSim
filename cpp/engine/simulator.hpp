#pragma once

#include "engine_input.hpp"
#include "event_log.hpp"
#include "simulation_state.hpp"

#include <cstdint>
#include <random>

/// Owns loaded input, mutable state, RNG, and event log. Call run_ticks to advance simulation time.
class Simulator {
public:
    /// Validates input structurally, then initializes state and RNG.
    explicit Simulator(EngineInput input);

    [[nodiscard]] const EngineInput& input() const noexcept { return input_; }
    [[nodiscard]] const SimulationState& state() const noexcept { return state_; }
    [[nodiscard]] const EventLog& events() const noexcept { return events_; }
    [[nodiscard]] std::uint64_t current_tick() const noexcept { return current_tick_; }

    /// Advances the simulation clock by n ticks.
    void run_ticks(std::uint64_t n);

private:
    EngineInput input_{};
    SimulationState state_{};
    EventLog events_{};
    std::mt19937_64 rng_{};
    std::uint64_t current_tick_{0};
};
