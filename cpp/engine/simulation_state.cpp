#include "simulation_state.hpp"

SimulationState::SimulationState(const EngineInput& in) {
    pack_state.assign(in.pack_initial_state.begin(), in.pack_initial_state.end());
    pack_location_id.assign(in.pack_initial_location_id.begin(), in.pack_initial_location_id.end());
    pack_market_id.assign(in.pack_initial_market_id.begin(), in.pack_initial_market_id.end());
}
