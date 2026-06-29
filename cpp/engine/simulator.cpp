#include "simulator.hpp"

#include "engine_input_validate.hpp"
#include "engine_input_make_view.hpp"
#include "enums.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <limits>
#include <stdexcept>

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

inline std::uint8_t pack_flag_mask(SimulationState::PackFlag flag) {
    return static_cast<std::uint8_t>(flag);
}

inline bool pack_has_flag(std::uint8_t flags, SimulationState::PackFlag flag) {
    return (flags & pack_flag_mask(flag)) != 0u;
}

inline void pack_set_flag(std::uint8_t& flags, SimulationState::PackFlag flag) {
    flags = static_cast<std::uint8_t>(flags | pack_flag_mask(flag));
}

inline void pack_clear_flag(std::uint8_t& flags, SimulationState::PackFlag flag) {
    flags = static_cast<std::uint8_t>(flags & static_cast<std::uint8_t>(~pack_flag_mask(flag)));
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

    const std::size_t n_locations = static_cast<std::size_t>(input_.n_locations);
    poisson_dist_by_loc_.resize(n_locations);
    for (std::size_t i = 0; i < n_locations; ++i) {
        const float lambda = input_.location_demand_poisson_lambda[i];
        poisson_dist_by_loc_[i] = std::poisson_distribution<std::uint32_t>(lambda);
    }
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
    sync_pack_registry(pack_id);
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
            sync_pack_registry(pack_id);
        }
    } else if (st == PACK_STATE::DECOMISSIONED) {
        if (bernoulli(rp)) {
            state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::ACTIVE);
            events_.push(tick, pack_id, EventType::REACTIVATE, loc, k_event_no_location);
            sync_pack_registry(pack_id);
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

std::uint32_t Simulator::lookup_next_edge(std::uint32_t src_loc, std::uint32_t dst_loc) const {
    const std::size_t s = static_cast<std::size_t>(src_loc);
    const auto beg = input_.location_route_offset[s];
    const auto end = input_.location_route_offset[s + 1];
    for (std::uint32_t i = beg; i < end; ++i) {
        if (input_.location_route_dst_location_id[static_cast<std::size_t>(i)] == dst_loc) {
            return input_.location_route_next_edge_id[static_cast<std::size_t>(i)];
        }
    }
    throw std::runtime_error("no route from src location to destination location");
}

std::uint32_t Simulator::pick_pack_for_shipment(std::uint32_t src_loc) const {
    std::uint32_t pid = state_.location_pack_head[static_cast<std::size_t>(src_loc)];
    while (pid != SimulationState::k_no_pack_id) {
        const std::size_t pi = static_cast<std::size_t>(pid);
        const auto st = static_cast<PACK_STATE>(state_.pack_state[pi]);
        const bool movable = (st == PACK_STATE::UPLOADED || st == PACK_STATE::ACTIVE);
        const bool reserved = pack_has_flag(state_.pack_flags[pi], SimulationState::PackFlag::RESERVED);
        if (movable && !reserved) return pid;
        pid = state_.pack_next[pi];
    }
    return SimulationState::k_no_pack_id;
}

std::uint32_t Simulator::pick_pool_pack_for_activation(std::uint32_t src_loc) const {
    std::uint32_t pid = state_.location_pack_head[static_cast<std::size_t>(src_loc)];
    while (pid != SimulationState::k_no_pack_id) {
        const std::size_t pi = static_cast<std::size_t>(pid);
        const auto st = static_cast<PACK_STATE>(state_.pack_state[pi]);
        const bool is_pool_pack = (st == PACK_STATE::DECOMISSIONED);
        const bool reserved = pack_has_flag(state_.pack_flags[pi], SimulationState::PackFlag::RESERVED);
        const bool in_transit = pack_has_flag(state_.pack_flags[pi], SimulationState::PackFlag::IN_TRANSIT);
        if (is_pool_pack && !reserved && !in_transit) return pid;
        pid = state_.pack_next[pi];
    }
    return SimulationState::k_no_pack_id;
}

void Simulator::schedule_pack_hop(std::uint32_t pack_id, std::uint32_t edge_id, std::uint32_t final_dst, std::uint64_t now_tick) {
    const std::uint64_t due = now_tick + static_cast<std::uint64_t>(
        input_.edge_lead_time_ticks[static_cast<std::size_t>(edge_id)]
    );
    ship_due_tick_.push_back(due);
    ship_pack_id_.push_back(pack_id);
    ship_edge_id_.push_back(edge_id);
    ship_final_dst_loc_.push_back(final_dst);
}

void Simulator::execute_due_shipments(std::uint64_t tick) {
    std::size_t write = 0;
    for (std::size_t i = 0; i < ship_due_tick_.size(); ++i) {
        if (ship_due_tick_[i] > tick) {
            if (write != i) {
                ship_due_tick_[write] = ship_due_tick_[i];
                ship_pack_id_[write] = ship_pack_id_[i];
                ship_edge_id_[write] = ship_edge_id_[i];
                ship_final_dst_loc_[write] = ship_final_dst_loc_[i];
            }
            ++write;
            continue;
        }

        const std::uint32_t pack_id = ship_pack_id_[i];
        const std::uint32_t edge_id = ship_edge_id_[i];
        const std::uint32_t final_dst = ship_final_dst_loc_[i];
        const std::size_t pi = static_cast<std::size_t>(pack_id);

        const std::uint32_t from = state_.pack_location_id[pi];
        const std::uint32_t to = input_.edge_dst_location_id[static_cast<std::size_t>(edge_id)];

        state_.relink_pack_location(pack_id, from, to);
        const std::uint32_t prod = input_.pack_product_id[pi];
        const std::size_t from_slot = state_.lp_index(from, prod);
        const std::size_t to_slot = state_.lp_index(to, prod);
        if (state_.location_on_hand[from_slot] > 0) {
            --state_.location_on_hand[from_slot];
        }
        ++state_.location_on_hand[to_slot];
        state_.pack_location_id[pi] = to;
        state_.pack_market_id[pi] = input_.location_market_id[static_cast<std::size_t>(to)];
        events_.push(tick, pack_id, EventType::MOVE, from, to);

        if (to == final_dst) {
            pack_clear_flag(state_.pack_flags[pi], SimulationState::PackFlag::IN_TRANSIT);
            pack_clear_flag(state_.pack_flags[pi], SimulationState::PackFlag::RESERVED);
            state_.pack_route_dst_location_id[pi] = SimulationState::k_no_location_id;
            on_pack_arrival(pack_id, to, tick);
        } else {
            const std::uint32_t next_edge = lookup_next_edge(to, final_dst);
            schedule_pack_hop(pack_id, next_edge, final_dst, tick);
        }

        sync_pack_registry(pack_id);
    }

    ship_due_tick_.resize(write);
    ship_pack_id_.resize(write);
    ship_edge_id_.resize(write);
    ship_final_dst_loc_.resize(write);
}

void Simulator::add_product_demand(std::uint32_t loc, std::uint32_t demand_units) {
    if (demand_units == 0) return;
    const int n_prod = state_.n_products;
    if (n_prod <= 1) {
        state_.location_backlog[state_.lp_index(loc, 0)] += demand_units;
        return;
    }
    // Deterministic proportional split across products by demand weight (no RNG draws, so the
    // engine's random stream -- movement/decommission/poisson -- is unaffected by the product
    // dimension). The integer remainder is handed out low-index first.
    double total_w = 0.0;
    for (int p = 0; p < n_prod; ++p) total_w += input_.product_demand_weight[static_cast<std::size_t>(p)];
    std::uint32_t remaining = demand_units;
    if (total_w > 0.0) {
        for (int p = 0; p < n_prod; ++p) {
            const double w = input_.product_demand_weight[static_cast<std::size_t>(p)];
            const auto share = static_cast<std::uint32_t>(
                static_cast<double>(demand_units) * w / total_w);
            if (share > 0) {
                state_.location_backlog[state_.lp_index(loc, static_cast<std::uint32_t>(p))] += share;
                remaining -= share;
            }
        }
    }
    for (int p = 0; remaining > 0; p = (p + 1) % n_prod, --remaining) {
        state_.location_backlog[state_.lp_index(loc, static_cast<std::uint32_t>(p))] += 1u;
    }
}

void Simulator::apply_demand_policy(std::uint32_t loc, std::uint64_t /*tick*/) {
    const std::size_t li = static_cast<std::size_t>(loc);
    const std::uint8_t pid = input_.location_demand_policy_id[li];
    if (pid == 0) return;
    if (pid == 1) {
        const std::int32_t rate = input_.location_demand_const_rate[li];
        if (rate > 0) {
            add_product_demand(loc, static_cast<std::uint32_t>(rate));
        }
        return;
    }
    if (pid == 2) {
        const float lambda = input_.location_demand_poisson_lambda[li];
        if (lambda > 0.0f) {
            const std::uint32_t demand_units = poisson_dist_by_loc_[li](rng_);
            add_product_demand(loc, demand_units);
        }
        return;
    }
    throw std::runtime_error("unknown demand policy id");
}

void Simulator::apply_supply_policy(
    std::uint32_t loc,
    std::uint64_t tick,
    std::vector<std::uint32_t>& edge_remaining_capacity
) {
    const std::size_t li = static_cast<std::size_t>(loc);
    const std::uint8_t pid = input_.location_supply_policy_id[li];
    if (pid == 0) return;
    if (pid == 1) {
        std::uint32_t cap = input_.location_supply_capacity_per_tick[li];
        std::int32_t target_S_i32 = input_.location_order_up_to_S[li];
        if (target_S_i32 <= 0) {
            target_S_i32 = input_.location_base_stock_level[li];
        }
        if (target_S_i32 <= 0) return;

        const auto target_S = static_cast<std::uint64_t>(target_S_i32);
        // Inventory position aggregates over products: the (s,S) policy decides reorder volume
        // at the location level; the product mix is determined by which packs are picked.
        const auto on_hand = state_.aggregate_for_location(state_.location_on_hand, loc);
        const auto pipeline = state_.aggregate_for_location(state_.location_pipeline_outstanding, loc);
        const auto inventory_position = on_hand + pipeline;
        if (inventory_position >= target_S) return;
        std::uint64_t remaining_to_order = target_S - inventory_position;
        const std::uint32_t preferred_edge = input_.location_preferred_supplier_edge_id[li];
        if (preferred_edge == std::numeric_limits<std::uint32_t>::max()) return;

        const std::uint32_t dst = input_.edge_dst_location_id[static_cast<std::size_t>(preferred_edge)];
        while (cap > 0 && remaining_to_order > 0) {
            const std::uint32_t pack_id = pick_pack_for_shipment(loc);
            if (pack_id == SimulationState::k_no_pack_id) break;
            const std::uint32_t first_edge = lookup_next_edge(loc, dst);
            const std::size_t first_edge_i = static_cast<std::size_t>(first_edge);
            if (edge_remaining_capacity[first_edge_i] == 0) break;

            const std::size_t pi = static_cast<std::size_t>(pack_id);
            const std::uint32_t prod = input_.pack_product_id[pi];
            pack_set_flag(state_.pack_flags[pi], SimulationState::PackFlag::RESERVED);
            pack_set_flag(state_.pack_flags[pi], SimulationState::PackFlag::IN_TRANSIT);
            state_.pack_route_dst_location_id[pi] = dst;
            schedule_pack_hop(pack_id, first_edge, dst, tick);
            --edge_remaining_capacity[first_edge_i];
            ++state_.location_pipeline_outstanding[state_.lp_index(dst, prod)];

            --cap;
            --remaining_to_order;
        }
        return;
    }
    if (pid == 2) {
        push_fanout(loc, tick, edge_remaining_capacity);
        return;
    }
    throw std::runtime_error("unknown supply policy id");
}

void Simulator::push_fanout(
    std::uint32_t loc,
    std::uint64_t tick,
    std::vector<std::uint32_t>& edge_remaining_capacity
) {
    const std::size_t li = static_cast<std::size_t>(loc);
    const std::uint32_t cap = input_.location_supply_capacity_per_tick[li];
    if (cap == 0) return;

    // Gather outgoing edges with gravity weights (capacity / (1 + cost)): high-capacity,
    // low-cost lanes carry proportionally more flow. Multi-edge fan-out replaces the old
    // single preferred-supplier lane, so flow naturally spreads across downstream neighbors
    // (removing the single-chain artifact) instead of funneling through one edge.
    const std::uint32_t beg = input_.location_out_edge_offset[li];
    const std::uint32_t end = input_.location_out_edge_offset[li + 1];
    if (beg >= end) return;

    struct OutEdge { float weight; std::uint32_t edge_id; };
    std::vector<OutEdge> edges;
    edges.reserve(static_cast<std::size_t>(end - beg));
    for (std::uint32_t i = beg; i < end; ++i) {
        const std::uint32_t eid = input_.location_out_edge_id[static_cast<std::size_t>(i)];
        const float cost = input_.edge_cost[static_cast<std::size_t>(eid)];
        const std::uint32_t ecap = input_.edge_capacity[static_cast<std::size_t>(eid)];
        edges.push_back(OutEdge{static_cast<float>(ecap) / (1.0f + cost), eid});
    }
    // Deterministic ordering: heaviest lane first, ties broken by edge id.
    std::sort(edges.begin(), edges.end(), [](const OutEdge& a, const OutEdge& b) {
        if (a.weight != b.weight) return a.weight > b.weight;
        return a.edge_id < b.edge_id;
    });

    // Interleave the per-tick capacity across the weight-sorted lanes (one pack per lane,
    // cycling): the heaviest lane is served first each round, but every lane gets a share even
    // when only a few packs are available -- so flow genuinely fans out across downstream
    // neighbors rather than funneling into the single dominant lane.
    const std::size_t m = edges.size();
    std::uint32_t remaining = cap;
    std::size_t idx = 0;
    std::size_t consecutive_full = 0;
    while (remaining > 0 && consecutive_full < m) {
        const std::uint32_t edge_id = edges[idx].edge_id;
        const std::size_t edge_i = static_cast<std::size_t>(edge_id);
        if (edge_remaining_capacity[edge_i] == 0) {
            idx = (idx + 1) % m;
            ++consecutive_full;
            continue;
        }
        bool from_pool = false;
        std::uint32_t pack_id = pick_pool_pack_for_activation(loc);
        if (pack_id != SimulationState::k_no_pack_id) {
            from_pool = true;
        } else {
            pack_id = pick_pack_for_shipment(loc);
        }
        if (pack_id == SimulationState::k_no_pack_id) return;  // no shippable stock left

        const std::uint32_t dst = input_.edge_dst_location_id[edge_i];
        const std::size_t pi = static_cast<std::size_t>(pack_id);
        if (from_pool) {
            state_.pack_state[pi] = static_cast<std::uint8_t>(PACK_STATE::UPLOADED);
        }
        const std::uint32_t prod = input_.pack_product_id[pi];
        pack_set_flag(state_.pack_flags[pi], SimulationState::PackFlag::RESERVED);
        pack_set_flag(state_.pack_flags[pi], SimulationState::PackFlag::IN_TRANSIT);
        state_.pack_route_dst_location_id[pi] = dst;

        schedule_pack_hop(pack_id, edge_id, dst, tick);
        --edge_remaining_capacity[edge_i];
        ++state_.location_pipeline_outstanding[state_.lp_index(dst, prod)];
        sync_pack_registry(pack_id);

        --remaining;
        consecutive_full = 0;
        idx = (idx + 1) % m;
    }
}

void Simulator::on_pack_arrival(std::uint32_t pack_id, std::uint32_t to_loc, std::uint64_t /*tick*/) {
    const std::size_t li = static_cast<std::size_t>(to_loc);
    const std::size_t pi = static_cast<std::size_t>(pack_id);
    const std::uint32_t prod = input_.pack_product_id[pi];

    const std::size_t prod_slot = state_.lp_index(to_loc, prod);
    if (state_.location_pipeline_outstanding[prod_slot] > 0) {
        --state_.location_pipeline_outstanding[prod_slot];
    }

    const ORG_TYPE ot = static_cast<ORG_TYPE>(state_.location_org_type[li]);
    if (ot != ORG_TYPE::LOCAL_ORG) return;

    // Product-aware fulfillment: satisfy this product's backlog first; on stockout, consult
    // the ATC substitute list (substitution-on-stockout) before giving up.
    if (state_.location_backlog[prod_slot] > 0) {
        --state_.location_backlog[prod_slot];
        return;
    }
    const std::size_t beg = input_.product_substitute_offset[static_cast<std::size_t>(prod)];
    const std::size_t end = input_.product_substitute_offset[static_cast<std::size_t>(prod) + 1];
    for (std::size_t k = beg; k < end; ++k) {
        const std::uint32_t sub = input_.product_substitute_id[k];
        const std::size_t sub_slot = state_.lp_index(to_loc, sub);
        if (state_.location_backlog[sub_slot] > 0) {
            --state_.location_backlog[sub_slot];
            return;
        }
    }
}

void Simulator::run_location_phase(std::uint64_t tick) {
    for (std::uint32_t loc = 0; loc < static_cast<std::uint32_t>(input_.n_locations); ++loc) {
        apply_demand_policy(loc, tick);
    }
}

void Simulator::run_supply_phase(std::uint64_t tick) {
    std::vector<std::uint32_t> edge_remaining_capacity(
        input_.edge_capacity.begin(),
        input_.edge_capacity.end()
    );
    for (std::uint32_t loc = 0; loc < static_cast<std::uint32_t>(input_.n_locations); ++loc) {
        apply_supply_policy(loc, tick, edge_remaining_capacity);
    }
}

void Simulator::run_shipment_phase(std::uint64_t tick) {
    execute_due_shipments(tick);
}

void Simulator::run_pack_behavior_phase(std::uint64_t tick) {
    const std::uint32_t n_packs = static_cast<std::uint32_t>(input_.n_packs);
    for (std::uint32_t pack_id = 0; pack_id < n_packs; ++pack_id) {
        const std::size_t pi = static_cast<std::size_t>(pack_id);
        const std::uint32_t loc = state_.pack_location_id[pi];
        apply_location_behavior(pack_id, loc, tick);
        sync_pack_registry(pack_id);
    }
}

void Simulator::apply_end_of_tick_penalty() {
    const std::size_t n_locations = static_cast<std::size_t>(input_.n_locations);
    for (std::size_t li = 0; li < n_locations; ++li) {
        const std::uint8_t pid = input_.location_penalty_policy_id[li];
        if (pid == 0) continue;
        if (pid == 1) {
            const float per_unit = input_.location_unfulfilled_unit_penalty[li];
            if (per_unit <= 0.0f) continue;
            // Penalty is charged on total unmet demand at the location (all products).
            const std::uint64_t backlog = state_.aggregate_for_location(
                state_.location_backlog, static_cast<std::uint32_t>(li));
            if (backlog == 0) continue;
            state_.location_cum_unfulfilled_penalty[li] +=
                static_cast<double>(per_unit) * static_cast<double>(backlog);
            continue;
        }
        throw std::runtime_error("unknown penalty policy id");
    }
}

void Simulator::sync_pack_registry(std::uint32_t pack_id) {
    state_.sync_registry_from_physical(pack_id);
}

void Simulator::run_ticks(std::uint64_t n) {
    for(std::uint64_t step = 0; step < n; ++step) {
        const std::uint64_t tick = current_tick_;
        run_location_phase(tick);
        run_supply_phase(tick);
        run_shipment_phase(tick);
        run_pack_behavior_phase(tick);
        apply_end_of_tick_penalty();
        ++current_tick_;
    }
}
