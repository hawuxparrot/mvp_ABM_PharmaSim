#pragma once

#include "engine_input.hpp"
#include "engine_input_view.hpp"
#include "event_log.hpp"
#include "simulation_state.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

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

    std::vector<std::uint64_t> ship_due_tick_;
    std::vector<std::uint32_t> ship_pack_id_;
    std::vector<std::uint32_t> ship_edge_id_;
    std::vector<std::uint32_t> ship_final_dst_loc_;

   

    void process_pack_tick(std::uint32_t pack_id, std::uint64_t tick);
    void apply_location_behavior(std::uint32_t pack_id, std::uint32_t loc, std::uint64_t tick);
    bool attempt_movement(std::uint32_t pack_id, std::uint32_t loc);
    bool has_outgoing_edge(std::uint32_t loc) const;
    void move_pack(std::uint32_t pack_id, std::uint32_t from_loc, std::uint64_t tick);
    void apply_post_movement_transition(std::uint32_t pack_id, std::uint32_t to_loc);
    void sync_pack_registry(std::uint32_t pack_id);

    void run_location_phase(std::uint64_t tick);
    void run_supply_phase(std::uint64_t tick);
    void run_shipment_phase(std::uint64_t tick);
    void run_pack_behavior_phase(std::uint64_t tick);

    void apply_demand_policy(std::uint32_t loc, std::uint64_t tick);
    void apply_supply_policy(
        std::uint32_t loc,
        std::uint64_t tick,
        std::vector<std::uint32_t>& edge_remaining_capacity
    );

    void schedule_pack_hop(std::uint32_t pack_id, std::uint32_t edge_id, std::uint32_t final_dst, std::uint64_t curr_tick);
    void execute_due_shipments(std::uint64_t tick);
    std::uint32_t lookup_next_edge(std::uint32_t src_loc, std::uint32_t dst_loc) const;
    std::uint32_t pick_pack_for_shipment(std::uint32_t src_loc) const;
    std::uint32_t pick_pool_pack_for_activation(std::uint32_t src_loc) const;
    void on_pack_arrival(std::uint32_t pack_id, std::uint32_t to_loc, std::uint64_t tick);


};
