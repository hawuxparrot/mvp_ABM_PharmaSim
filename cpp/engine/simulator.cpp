#include "simulator.hpp"

#include "engine_input_validate.hpp"
#include "engine_input_make_view.hpp"
#include "enums.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>

#include <limits>

namespace {

constexpr float k_move_probability = 0.25f;

inline std::uint64_t prob_to_threshold(double p) {
    if (!(p > 0.0)) return 0.0;
    if (p >= 1.0) return std::numeric_limits<std::uint64_t>::max();

    const auto space = (static_cast<unsigned __int128>(1) << 64);
    auto t = static_cast<unsigned __int128>(static_cast<long double>(p) * static_cast<long double>(space));
    if (t >= space) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(t);
}

const std::uint64_t k_move_threshold = prob_to_threshold(static_cast<double>(k_move_probability));

bool is_terminal_org_for_movement(ORG_TYPE ot) {
    return ot == ORG_TYPE::LOCAL_ORG || ot == ORG_TYPE::NMVO || ot == ORG_TYPE::EMVO;
}

/// Picks an outgoing edge id from ``src_loc`` using weights ``capacity / (1 + cost)``.
/// Precondition: ``src_loc`` has at least one outgoing edge.
std::uint32_t pick_outgoing_edge(const EngineInputView& in, std::uint32_t src_loc, std::mt19937_64& rng) {
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

void Simulator::init(std::shared_ptr<const void> owner, EngineInputView input) {
    validate_engine_input_or_throw(input);
    input_ = std::move(input);
    owner_ = std::move(owner);
    state_ = SimulationState(input_);
    rng_.seed(static_cast<std::uint64_t>(input_.seed));
}

Simulator::Simulator(std::shared_ptr<const void> owner, EngineInputView input) {
    init(std::move(owner), std::move(input));
}

Simulator::Simulator(EngineInput input) {
    auto owned = std::make_shared<EngineInput>(std::move(input));
    EngineInputView view = make_view(*owned);
    std::shared_ptr<const void> opaque = owned;
    init(std::move(opaque), std::move(view));
}

bool Simulator::bernoulli(float p) {
    if (p <= 0.f) return false;
    if (p >= 1.f) return true;
    return dist_(rng_) < p;
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

void Simulator::process_pack_tick(std::uint32_t pack_id, std::uint64_t tick) {
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const std::uint32_t loc = state_.pack_location_id[pi];
    
    apply_location_behavior(pack_id, loc, tick);
    attempt_movement(pack_id, loc);    
}

void Simulator::apply_location_behavior(std::uint32_t pack_id, std::uint32_t loc, std::uint64_t tick) {
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const std::size_t li = static_cast<std::size_t>(loc);
    
    if (input_.location_has_behavior[li] == 0) return;

    const float vp = input_.location_verify_prob[li];
    const float dp = input_.location_decommission_prob[li];
    const float rp = input_.location_reactivate_prob[li];

    if (bernoulli(vp)) {
        events_.push(tick, pack_id, EventType::VERIFY, loc, k_event_no_location);
    }
    
    const auto st = static_cast<PACK_STATE>(state_.pack_state[pi]);
    if (st == PACK_STATE::ACTIVE) {
        if (bernoulli(dp)) {
            state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::DECOMISSIONED);
            events_.push(tick, pack_id, EventType::DECOMMISSION, loc, k_event_no_location);
        }
    } else if (st == PACK_STATE::DECOMISSIONED) {
        if (bernoulli(rp)) {
            state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
            events_.push(tick, pack_id, EventType::REACTIVATE, loc, k_event_no_location);
        }
    }
}

bool Simulator::has_outgoing_edge(std::uint32_t loc) const {
    const std::size_t li = static_cast<std::size_t>(loc);
    const std::uint32_t beg = input_.location_out_edge_offset[li];
    const std::uint32_t end = input_.location_out_edge_offset[li + 1];
    return beg < end;
}

bool Simulator::attempt_movement(std::uint32_t pack_id, std::uint32_t loc) {
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const std::size_t li = static_cast<std::size_t>(loc);

    const auto st = static_cast<PACK_STATE>(state_.pack_state[pi]);
    const bool movable_state = (st == PACK_STATE::UPLOADED || st == PACK_STATE::ACTIVE);  // Assume only uploaded and active packs can move, may not be the case for fraud detection
    const ORG_TYPE org_at_loc = static_cast<ORG_TYPE>(state_.location_org_type[li]);
    
    if (!movable_state || is_terminal_org_for_movement(org_at_loc) || (rng_() >= k_move_threshold) || !has_outgoing_edge(loc)) {
        return false;
    }

    move_pack(pack_id, loc, current_tick_);
    sync_pack_registry(pack_id);
    return true;
}

void Simulator::move_pack(std::uint32_t pack_id, std::uint32_t from_loc, std::uint64_t tick) {
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const std::uint32_t edge_id = pick_outgoing_edge(input_, from_loc, rng_);
    const std::uint32_t dst = input_.edge_dst_location_id[static_cast<std::size_t>(edge_id)];
    
    state_.pack_location_id[pi] = dst;
    state_.pack_market_id[pi] = input_.location_market_id[static_cast<std::size_t>(dst)];
    events_.push(tick, pack_id, EventType::MOVE, from_loc, dst);
    
    apply_post_movement_transition(pack_id, dst);
}

void Simulator::apply_post_movement_transition(std::uint32_t pack_id, std::uint32_t to_loc) {
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const ORG_TYPE org_at_dst = static_cast<ORG_TYPE>(state_.location_org_type[static_cast<std::size_t>(to_loc)]);

    if (static_cast<PACK_STATE>(state_.pack_state[pi]) == PACK_STATE::UPLOADED && org_at_dst == ORG_TYPE::WHOLESALER) {
        state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
    }
}

void Simulator::sync_pack_registry(std::uint32_t pack_id) {
    state_.sync_registry_from_physical(pack_id);
}

void Simulator::run_ticks(std::uint64_t n) {
    const int n_packs = input_.n_packs;
    for (std::uint64_t step = 0; step < n; ++step) {
        const std::uint64_t tick = current_tick_;
        for (std::uint32_t pack_id = 0; pack_id < n_packs; ++pack_id) {
            process_pack_tick(pack_id, tick);
        }
        ++current_tick_;
    }
}
