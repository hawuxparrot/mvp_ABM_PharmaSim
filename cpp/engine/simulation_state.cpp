#include "simulation_state.hpp"
#include "engine_input_view.hpp"

SimulationState::SimulationState(const EngineInputView& in) {
    pack_state.assign(in.pack_initial_state.begin(), in.pack_initial_state.end());
    pack_location_id.assign(in.pack_initial_location_id.begin(), in.pack_initial_location_id.end());
    pack_market_id.assign(in.pack_initial_market_id.begin(), in.pack_initial_market_id.end());

    registry_pack_state = pack_state;
    registry_pack_location_id = pack_location_id;
    registry_pack_market_id = pack_market_id;

    location_org_type.resize(static_cast<std::size_t>(in.n_locations));
    for (int loc = 0; loc < in.n_locations; ++loc) {
        const std::uint32_t org_id = in.location_org_id[static_cast<std::size_t>(loc)];
        location_org_type[static_cast<std::size_t>(loc)] = in.org_type[static_cast<std::size_t>(org_id)];
    }
}

SimulationState::SimulationState(const EngineInput& in) {
    pack_state.assign(in.pack_initial_state.begin(), in.pack_initial_state.end());
    pack_location_id.assign(in.pack_initial_location_id.begin(), in.pack_initial_location_id.end());
    pack_market_id.assign(in.pack_initial_market_id.begin(), in.pack_initial_market_id.end());

    registry_pack_state = pack_state;
    registry_pack_location_id = pack_location_id;
    registry_pack_market_id = pack_market_id;

    location_org_type.resize(static_cast<std::size_t>(in.n_locations));
    for (int loc = 0; loc < in.n_locations; ++loc) {
        const std::uint32_t org_id = in.location_org_id[static_cast<std::size_t>(loc)];
        location_org_type[static_cast<std::size_t>(loc)] = in.org_type[static_cast<std::size_t>(org_id)];
    }
}

void SimulationState::sync_registry_from_physical(const std::uint32_t pack_index) {
    const std::size_t i = static_cast<std::size_t>(pack_index);
    registry_pack_state[i] = pack_state[i];
    registry_pack_location_id[i] = pack_location_id[i];
    registry_pack_market_id[i] = pack_market_id[i];
}
