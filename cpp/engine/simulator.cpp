#include "simulator.hpp"

#include "engine_input_validate.hpp"

Simulator::Simulator(EngineInput input) {
    // Validate before copying pack columns into SimulationState (structural safety only).
    validate_engine_input_or_throw(input);
    input_ = std::move(input);
    state_ = SimulationState(input_);
    rng_.seed(static_cast<std::uint64_t>(input_.seed));
}

void Simulator::run_ticks(std::uint64_t n) {
    current_tick_ += n;
    // Future: per-tick FSM, RNG draws, append to events_.
}
