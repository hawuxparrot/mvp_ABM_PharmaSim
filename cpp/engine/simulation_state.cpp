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

    location_on_hand.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_backlog.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_pipeline_outstanding.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_forecast.assign(static_cast<std::size_t>(in.n_locations), 0.f);
    location_last_order_tick.assign(static_cast<std::size_t>(in.n_locations), 0ull);
    location_cum_unfulfilled_penalty.assign(static_cast<std::size_t>(in.n_locations), 0.0);
    for (int loc = 0; loc < in.n_locations; ++loc) {
        const std::size_t li = static_cast<std::size_t>(loc);
        const auto clamp_u32 = [](std::int32_t v) -> std::uint32_t {
            return v < 0 ? 0u : static_cast<std::uint32_t>(v);
        };
        location_on_hand[li] = clamp_u32(in.location_initial_on_hand[li]);
        location_backlog[li] = clamp_u32(in.location_initial_backlog[li]);
        location_pipeline_outstanding[li] = clamp_u32(in.location_initial_pipeline_outstanding[li]);
    }
    rebuild_location_pack_index(static_cast<std::uint32_t>(in.n_locations));
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

    location_on_hand.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_backlog.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_pipeline_outstanding.assign(static_cast<std::size_t>(in.n_locations), 0u);
    location_forecast.assign(static_cast<std::size_t>(in.n_locations), 0.f);
    location_last_order_tick.assign(static_cast<std::size_t>(in.n_locations), 0ull);
    location_cum_unfulfilled_penalty.assign(static_cast<std::size_t>(in.n_locations), 0.0);
    for (int loc = 0; loc < in.n_locations; ++loc) {
        const std::size_t li = static_cast<std::size_t>(loc);
        const auto clamp_u32 = [](std::int32_t v) -> std::uint32_t {
            return v < 0 ? 0u : static_cast<std::uint32_t>(v);
        };
        location_on_hand[li] = clamp_u32(in.location_initial_on_hand[li]);
        location_backlog[li] = clamp_u32(in.location_initial_backlog[li]);
        location_pipeline_outstanding[li] = clamp_u32(in.location_initial_pipeline_outstanding[li]);
    }
    rebuild_location_pack_index(static_cast<std::uint32_t>(in.n_locations));
}

void SimulationState::sync_registry_from_physical(const std::uint32_t pack_index) {
    const std::size_t i = static_cast<std::size_t>(pack_index);
    registry_pack_state[i] = pack_state[i];
    registry_pack_location_id[i] = pack_location_id[i];
    registry_pack_market_id[i] = pack_market_id[i];
}

void SimulationState::rebuild_location_pack_index(std::uint32_t n_locations) {
    location_pack_head.assign(n_locations, k_no_pack_id);
    pack_next.assign(pack_location_id.size(), k_no_pack_id);
    pack_prev.assign(pack_location_id.size(), k_no_pack_id);

    for (std::uint32_t pack_id = 0; pack_id < pack_location_id.size(); ++pack_id) {
        const std::uint32_t loc = pack_location_id[pack_id];
        const std::uint32_t next = location_pack_head[loc];
        pack_next[pack_id] = next;
        pack_prev[pack_id] = k_no_pack_id;
        if (next != k_no_pack_id) {
            pack_prev[next] = pack_id;
        }
        location_pack_head[loc] = pack_id;
    }

    pack_flags.assign(pack_location_id.size(), 0u);
    pack_route_dst_location_id.assign(pack_location_id.size(), k_no_location_id);
}

void SimulationState::unlink_pack_from_location(std::uint32_t pack_id, std::uint32_t loc) {
    const std::uint32_t prev = pack_prev[pack_id];
    const std::uint32_t next = pack_next[pack_id];
    if (prev == k_no_pack_id) {
        location_pack_head[loc] = next;
    } else {
        pack_next[prev] = next;
    }

    if (next != k_no_pack_id) {
        pack_prev[next] = prev;
    }

    pack_prev[pack_id] = k_no_pack_id;
    pack_next[pack_id] = k_no_pack_id;
}

void SimulationState::link_pack_to_location_head(std::uint32_t pack_id, std::uint32_t loc) {
    const std::uint32_t head = location_pack_head[loc];
    pack_next[pack_id] = head;
    pack_prev[pack_id] = k_no_pack_id;
    if (head != k_no_pack_id) {
        pack_prev[head] = pack_id;
    }
    location_pack_head[loc] = pack_id;
}

void SimulationState::relink_pack_location(std::uint32_t pack_id, std::uint32_t old_loc, std::uint32_t new_loc) {
    if (old_loc == new_loc) return;
    unlink_pack_from_location(pack_id, old_loc);
    link_pack_to_location_head(pack_id, new_loc);
}