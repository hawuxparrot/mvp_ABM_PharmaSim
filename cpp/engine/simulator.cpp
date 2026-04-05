#include "simulator.hpp"

#include "engine_input_validate.hpp"
#include "enums.hpp"

#include <cstddef>
#include <cstdint>
#include <random>

namespace {

constexpr float k_move_probability = 0.25f;

bool is_terminal_org_for_movement(ORG_TYPE ot) {
    return ot == ORG_TYPE::LOCAL_ORG || ot == ORG_TYPE::NMVO || ot == ORG_TYPE::EMVO;
}

bool bernoulli(std::mt19937_64& rng, float p) {
    if (p <= 0.f) {
        return false;
    }
    if (p >= 1.f) {
        return true;
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng) < p;
}

/// Picks an outgoing edge id from ``src_loc`` using weights ``capacity / (1 + cost)``.
/// Precondition: ``src_loc`` has at least one outgoing edge.
std::uint32_t pick_outgoing_edge(const EngineInput& in, std::uint32_t src_loc, std::mt19937_64& rng) {
    const std::size_t sl = static_cast<std::size_t>(src_loc);
    const std::uint32_t beg = in.location_out_edge_offset[sl];
    const std::uint32_t end = in.location_out_edge_offset[sl + 1];

    float total_w = 0.f;
    for (std::uint32_t i = beg; i < end; ++i) {
        const std::uint32_t eid = in.location_out_edge_id[static_cast<std::size_t>(i)];
        const float cost = in.edge_cost[static_cast<std::size_t>(eid)];
        const std::uint32_t cap = in.edge_capacity[static_cast<std::size_t>(eid)];
        total_w += static_cast<float>(cap) / (1.0f + cost);
    }

    if (total_w <= 0.f) {
        std::uniform_int_distribution<std::uint32_t> dist(0, end - beg - 1);
        const std::uint32_t slot = beg + dist(rng);
        return in.location_out_edge_id[static_cast<std::size_t>(slot)];
    }

    std::uniform_real_distribution<float> u(0.0f, total_w);
    float r = u(rng);
    for (std::uint32_t i = beg; i < end; ++i) {
        const std::uint32_t eid = in.location_out_edge_id[static_cast<std::size_t>(i)];
        const float cost = in.edge_cost[static_cast<std::size_t>(eid)];
        const std::uint32_t cap = in.edge_capacity[static_cast<std::size_t>(eid)];
        const float w = static_cast<float>(cap) / (1.0f + cost);
        r -= w;
        if (r <= 0.f) {
            return eid;
        }
    }
    return in.location_out_edge_id[static_cast<std::size_t>(end - 1)];
}

} // namespace

Simulator::Simulator(EngineInput input) {
    validate_engine_input_or_throw(input);
    input_ = std::move(input);
    state_ = SimulationState(input_);
    rng_.seed(static_cast<std::uint64_t>(input_.seed));
}

bool Simulator::registry_matches_physical() const noexcept {
    const SimulationState& s = state_;
    for (std::size_t i = 0; i < s.pack_state.size(); ++i) {
        if (s.registry_pack_state[i] != s.pack_state[i]) {
            return false;
        }
        if (s.registry_pack_location_id[i] != s.pack_location_id[i]) {
            return false;
        }
        if (s.registry_pack_market_id[i] != s.pack_market_id[i]) {
            return false;
        }
    }
    return true;
}

void Simulator::run_ticks(std::uint64_t n) {
    const int n_packs = input_.n_packs;

    for (std::uint64_t step = 0; step < n; ++step) {
        const std::uint64_t tick = current_tick_;

        for (int pack = 0; pack < n_packs; ++pack) {
            const std::uint32_t pid = static_cast<std::uint32_t>(pack);
            const std::size_t pi = static_cast<std::size_t>(pack);

            std::uint32_t loc = state_.pack_location_id[pi];
            const std::size_t li = static_cast<std::size_t>(loc);
            const ORG_TYPE org_at_loc = static_cast<ORG_TYPE>(state_.location_org_type[li]);

            // Behavior at start of tick (VERIFY / DECOMMISSION / REACTIVATE), then movement.
            if (input_.location_has_behavior[li] != 0) {
                const float vp = input_.location_verify_prob[li];
                const float dp = input_.location_decommission_prob[li];
                const float rp = input_.location_reactivate_prob[li];

                if (bernoulli(rng_, vp)) {
                    events_.push(tick, pid, EventType::VERIFY, loc, k_event_no_location);
                }

                const auto st = static_cast<PACK_STATE>(state_.pack_state[pi]);
                if (st == PACK_STATE::ACTIVE) {
                    if (bernoulli(rng_, dp)) {
                        state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::DECOMISSIONED);
                        events_.push(tick, pid, EventType::DECOMMISSION, loc, k_event_no_location);
                    }
                } else if (st == PACK_STATE::DECOMISSIONED) {
                    if (bernoulli(rng_, rp)) {
                        state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
                        events_.push(tick, pid, EventType::REACTIVATE, loc, k_event_no_location);
                    }
                }
            }

            const auto st_move = static_cast<PACK_STATE>(state_.pack_state[pi]);
            const bool movable_state =
                (st_move == PACK_STATE::UPLOADED || st_move == PACK_STATE::ACTIVE);
            if (!movable_state || is_terminal_org_for_movement(org_at_loc)) {
                state_.sync_registry_from_physical(pid);
                continue;
            }

            if (!bernoulli(rng_, k_move_probability)) {
                state_.sync_registry_from_physical(pid);
                continue;
            }

            const std::uint32_t beg = input_.location_out_edge_offset[li];
            const std::uint32_t end = input_.location_out_edge_offset[li + 1];
            if (beg >= end) {
                state_.sync_registry_from_physical(pid);
                continue;
            }

            const std::uint32_t edge_id = pick_outgoing_edge(input_, loc, rng_);
            const std::uint32_t dst =
                input_.edge_dst_location_id[static_cast<std::size_t>(edge_id)];
            const std::uint32_t from_loc = loc;

            state_.pack_location_id[pi] = dst;
            state_.pack_market_id[pi] = input_.location_market_id[static_cast<std::size_t>(dst)];
            events_.push(tick, pid, EventType::MOVE, from_loc, dst);

            const ORG_TYPE org_at_dst =
                static_cast<ORG_TYPE>(state_.location_org_type[static_cast<std::size_t>(dst)]);
            if (static_cast<PACK_STATE>(state_.pack_state[pi]) == PACK_STATE::UPLOADED
                && org_at_dst == ORG_TYPE::WHOLESALER) {
                state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
            }

            state_.sync_registry_from_physical(pid);
        }

        ++current_tick_;
    }
}
