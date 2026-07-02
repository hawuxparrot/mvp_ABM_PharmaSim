#include "simulation_state.hpp"
#include "engine_input_view.hpp"
#include "enums.hpp"

#include <algorithm>

namespace {

/// Build the per-serial NMVS records from the initial pack table + batch provisioning.
///
/// The manufacturer "upload" is modelled as: every serial present in the pack table is provisioned
/// to the intended markets of its batch, and starts Active (uploaded/active packs). Two physical
/// packs that share a serial id increment the same ``serial_pack_count`` -- the clone signal.
template <typename InputT>
void init_serial_registry(SimulationState& s, const InputT& in) {
    const auto n_serials = static_cast<std::size_t>(in.n_serials);
    s.registry_serial_state.assign(n_serials, static_cast<std::uint8_t>(PACK_STATE::DECOMISSIONED));
    s.registry_serial_reason.assign(n_serials, static_cast<std::uint8_t>(DECOMMISSION_REASON::DISPENSED));
    s.registry_serial_deadline_tick.assign(n_serials, 0ull);
    s.serial_pack_count.assign(n_serials, 0u);
    s.serial_markets.assign(n_serials, {});

    const std::size_t n_pack = static_cast<std::size_t>(in.n_packs);
    for (std::size_t i = 0; i < n_pack; ++i) {
        const std::uint32_t serial = in.pack_serial_id[i];
        const std::size_t si = static_cast<std::size_t>(serial);
        s.serial_pack_count[si] += 1u;

        // A serial provisioned by any uploaded/active pack is Active in the NMVS.
        const auto pst = static_cast<PACK_STATE>(in.pack_initial_state[i]);
        if (pst == PACK_STATE::UPLOADED || pst == PACK_STATE::ACTIVE) {
            s.registry_serial_state[si] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
        }

        // Provision the serial to its batch's intended markets.
        const std::uint32_t batch = in.pack_batch_id[i];
        const std::size_t beg = in.batch_intended_market_offset[static_cast<std::size_t>(batch)];
        const std::size_t end = in.batch_intended_market_offset[static_cast<std::size_t>(batch) + 1];
        for (std::size_t k = beg; k < end; ++k) {
            const std::uint32_t m = in.batch_intended_market_id[k];
            auto& markets = s.serial_markets[si];
            if (std::find(markets.begin(), markets.end(), m) == markets.end()) {
                markets.push_back(m);
            }
        }
    }
}

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

    init_serial_registry(*this, in);

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

    init_serial_registry(*this, in);

    location_org_type.resize(static_cast<std::size_t>(in.n_locations));
    for (int loc = 0; loc < in.n_locations; ++loc) {
        const std::uint32_t org_id = in.location_org_id[static_cast<std::size_t>(loc)];
        location_org_type[static_cast<std::size_t>(loc)] = in.org_type[static_cast<std::size_t>(org_id)];
    }

    init_location_state(*this, in);
    rebuild_location_pack_index(static_cast<std::uint32_t>(in.n_locations));
}

bool SimulationState::serial_market_is_provisioned(
    std::uint32_t serial_id, std::uint32_t market_id) const {
    const auto& markets = serial_markets[static_cast<std::size_t>(serial_id)];
    return std::find(markets.begin(), markets.end(), market_id) != markets.end();
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