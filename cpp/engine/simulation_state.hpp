#pragma once

#include "engine_input.hpp"
#include "engine_input_view.hpp"

#include <cstdint>
#include <vector>
#include <limits>

/// Mutable per-pack state for the simulation kernel (SoA). Initialized from ``EngineInput`` pack columns.
struct SimulationState {
    /// Physical ground truth (authoritative for the MVP kernel).
    std::vector<std::uint8_t> pack_state;
    std::vector<std::uint32_t> pack_location_id;
    std::vector<std::uint32_t> pack_market_id;

    /// NMVO/EMVO registry mirror (v1: identical copy each step; later decoupled).
    std::vector<std::uint8_t> registry_pack_state;
    std::vector<std::uint32_t> registry_pack_location_id;
    std::vector<std::uint32_t> registry_pack_market_id;

    /// ``org_type`` of the organization owning each location (by location id).
    std::vector<std::uint8_t> location_org_type;

    /// Number of products; per-(location, product) state is flattened as li * n_products + p.
    int n_products{0};

    // order system: per-(location, product), flattened row-major (location-major).
    std::vector<std::uint32_t> location_on_hand;                    // on_hand >= 0
    std::vector<std::uint32_t> location_backlog;                    // backlog >= 0
    std::vector<std::uint32_t> location_pipeline_outstanding;       // pipeline_outstanding >= 0
    // per-location aggregates (do not carry a product dimension)
    std::vector<float> location_forecast;
    std::vector<std::uint64_t> location_last_order_tick;
    std::vector<double> location_cum_unfulfilled_penalty;

    /// Flattened index for per-(location, product) state.
    [[nodiscard]] std::size_t lp_index(std::uint32_t loc, std::uint32_t product) const noexcept {
        return static_cast<std::size_t>(loc) * static_cast<std::size_t>(n_products)
             + static_cast<std::size_t>(product);
    }

    /// Sum a per-(location, product) vector over products for a single location.
    [[nodiscard]] std::uint64_t aggregate_for_location(
        const std::vector<std::uint32_t>& per_product, std::uint32_t loc) const noexcept {
        std::uint64_t total = 0;
        const std::size_t base = static_cast<std::size_t>(loc) * static_cast<std::size_t>(n_products);
        for (int p = 0; p < n_products; ++p) total += per_product[base + static_cast<std::size_t>(p)];
        return total;
    }

    /// Build a per-location vector by summing a per-(location, product) vector over products.
    [[nodiscard]] std::vector<std::uint32_t> aggregate_by_location(
        const std::vector<std::uint32_t>& per_product) const {
        const std::size_t n_loc = location_org_type.size();
        std::vector<std::uint32_t> out(n_loc, 0u);
        for (std::size_t li = 0; li < n_loc; ++li) {
            out[li] = static_cast<std::uint32_t>(
                aggregate_for_location(per_product, static_cast<std::uint32_t>(li)));
        }
        return out;
    }

    static constexpr std::uint32_t k_no_pack_id = std::numeric_limits<std::uint32_t>::max();
    static constexpr std::uint32_t k_no_location_id = std::numeric_limits<std::uint32_t>::max();
    
    enum class PackFlag : std::uint8_t {
        RESERVED = 1u << 0,
        IN_TRANSIT = 1u << 1,
    };
    
    std::vector<std::uint32_t> location_pack_head;          // per location linked list head
    std::vector<std::uint32_t> pack_next;                   // per pack next node in linked list
    std::vector<std::uint32_t> pack_prev;                   // per pack prev node in linked list
    std::vector<std::uint8_t> pack_flags;
    std::vector<std::uint32_t> pack_route_dst_location_id;
    
    void rebuild_location_pack_index(std::uint32_t n_locations);
    void unlink_pack_from_location(std::uint32_t pack_id, std::uint32_t loc);
    void link_pack_to_location_head(std::uint32_t pack_id, std::uint32_t loc);
    void relink_pack_location(std::uint32_t pack_id, std::uint32_t old_loc, std::uint32_t new_loc);

    SimulationState() = default;

    explicit SimulationState(const EngineInput& in);

    explicit SimulationState(const EngineInputView& in);

    void sync_registry_from_physical(std::uint32_t pack_index);
};
