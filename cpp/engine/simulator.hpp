#pragma once

#include "engine_input.hpp"
#include "engine_input_view.hpp"
#include "event_log.hpp"
#include "simulation_state.hpp"

#include <cstdint>
#include <memory>
#include <random>

/// Owns loaded input, mutable state, RNG, and event log. Call run_ticks to advance simulation time.
class Simulator {
public:
    /// Validates input structurally, then initializes state and RNG.
    Simulator(std::shared_ptr<const void> owner, EngineInputView input);
    
    explicit Simulator(EngineInput input);

    [[nodiscard]] const EngineInputView& input() const noexcept { return input_; }
    [[nodiscard]] const SimulationState& state() const noexcept { return state_; }
    [[nodiscard]] const EventLog& events() const noexcept { return events_; }
    [[nodiscard]] std::uint64_t current_tick() const noexcept { return current_tick_; }

    [[nodiscard]] std::size_t event_count() const noexcept { return events_.tick.size(); }
    [[nodiscard]] bool registry_matches_physical() const noexcept;
    [[nodiscard]] bool bernoulli(float p);

    /// Advances the simulation clock by n ticks.
    void run_ticks(std::uint64_t n);


private:
    void init(std::shared_ptr<const void> owner, EngineInputView input);

    EngineInputView input_{};
    std::shared_ptr<const void> owner_;
    SimulationState state_{};
    EventLog events_{};
    std::mt19937_64 rng_{};
    std::uniform_real_distribution<float> dist_{}; // [0.0f, 1.0f]
    std::uint64_t current_tick_{0};
   

    void process_pack_tick(std::uint32_t pack_id, std::uint64_t tick);
    void apply_location_behavior(std::uint32_t pack_id, std::uint32_t loc, std::uint64_t tick);
    bool attempt_movement(std::uint32_t pack_id, std::uint32_t loc);
    bool has_outgoing_edge(std::uint32_t loc) const;
    void move_pack(std::uint32_t pack_id, std::uint32_t from_loc, std::uint64_t tick);
    void apply_post_movement_transition(std::uint32_t pack_id, std::uint32_t to_loc);
    void sync_pack_registry(std::uint32_t pack_id);

};
