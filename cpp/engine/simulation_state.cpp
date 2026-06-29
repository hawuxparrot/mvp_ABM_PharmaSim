#include "simulation_state.hpp"
#include "engine_input_view.hpp"

namespace {

/// Shared per-(location, product) state initialization for both input flavors.
template <typename InputT>
void init_location_state(SimulationState& s, const InputT& in) {
    const std::size_t n_loc = static_cast<std::size_t>(in.n_locations);
    // Guarantee at least one product slot so the flattened layout (li * n_products + p) is
    // always valid even for degenerate product-less scenarios.
    s.n_products = in.n_products > 0 ? in.n_products : 1;
    const std::size_t n_prod = static_cast<std::size_t>(s.n_products);
    const std::size_t cells = n_loc * n_prod;

    s.location_on_hand.assign(cells, 0u);
    s.location_backlog.assign(cells, 0u);
    s.location_pipeline_outstanding.assign(cells, 0u);
    s.location_forecast.assign(n_loc, 0.f);
    s.location_last_order_tick.assign(n_loc, 0ull);
    s.location_cum_unfulfilled_penalty.assign(n_loc, 0.0);

    // Initial aggregate state is per-location in the ABI; seed it into product slot 0. This
    // preserves aggregate inventory/backlog semantics (sums over products) while the engine
    // now tracks the product dimension created by demand/shipments.
    const auto clamp_u32 = [](std::int32_t v) -> std::uint32_t {
        return v < 0 ? 0u : static_cast<std::uint32_t>(v);
    };
    for (std::size_t li = 0; li < n_loc; ++li) {
        const std::size_t slot0 = li * n_prod;
        s.location_on_hand[slot0] = clamp_u32(in.location_initial_on_hand[li]);
        s.location_backlog[slot0] = clamp_u32(in.location_initial_backlog[li]);
        s.location_pipeline_outstanding[slot0] = clamp_u32(in.location_initial_pipeline_outstanding[li]);
    }
}

}  // namespace

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

    init_location_state(*this, in);
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

    init_location_state(*this, in);
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