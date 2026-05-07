#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// Mirrors ``compiler.types.EngineInput`` (schema ``engine_input.v2``).
/// Location behavior (Option B): four vectors always have size ``n_locations``; use zeros when
/// no per-site policy applies (matches Python always-dense emission).
struct EngineInput {
    std::string schema_version;
    std::int64_t seed;

    int n_organizations;
    int n_locations;
    int n_products;
    int n_batches;
    int n_packs;
    int n_markets;
    int n_edges;

    std::vector<std::string> market_code;
    std::vector<std::uint8_t> org_type;
    std::vector<std::uint32_t> location_org_id;
    std::vector<std::uint32_t> location_market_id;
    std::vector<std::uint32_t> location_out_edge_offset;
    std::vector<std::uint32_t> location_out_edge_id;

    std::vector<std::uint32_t> edge_src_location_id;
    std::vector<std::uint32_t> edge_dst_location_id;
    std::vector<float> edge_cost;
    std::vector<std::uint32_t> edge_capacity;
    // order transport
    std::vector<std::uint16_t> edge_lead_time_ticks;

    std::vector<std::uint32_t> batch_product_id;
    std::vector<std::uint32_t> batch_manufacturer_org_id;
    std::vector<std::uint32_t> batch_intended_market_offset;
    std::vector<std::uint32_t> batch_intended_market_id;

    std::vector<std::uint32_t> pack_product_id;
    std::vector<std::uint32_t> pack_batch_id;
    std::vector<std::uint32_t> pack_initial_location_id;
    std::vector<std::uint32_t> pack_initial_market_id;
    std::vector<std::uint8_t> pack_initial_state;
    std::vector<std::string> pack_serial;

    std::vector<std::uint8_t> location_has_behavior;
    std::vector<float> location_verify_prob;
    std::vector<float> location_decommission_prob;
    std::vector<float> location_reactivate_prob;
    // location order/supply policy
    std::vector<std::uint8_t> location_demand_policy_id;
    std::vector<std::uint8_t> location_supply_policy_id;
    // initial aggregate state
    std::vector<std::int32_t> location_initial_on_hand;
    std::vector<std::int32_t> location_initial_backlog;
    std::vector<std::int32_t> location_initial_pipeline_outstanding;
    // demand policy params
    std::vector<std::int32_t> location_demand_const_rate;
    std::vector<std::int32_t> location_reorder_point_s;
    std::vector<std::int32_t> location_order_up_to_S;
    std::vector<std::int32_t> location_base_stock_level;
    std::vector<float> location_ewma_alpha;
    // supply policy params
    std::vector<std::uint32_t> location_supply_capacity_per_tick;
    std::vector<std::uint32_t> location_min_order_interval_ticks;
    // cost, penalty params
    std::vector<float> location_unfulfilled_unit_penalty;
    // supplier selection
    std::vector<std::uint32_t> location_preferred_supplier_edge_id;


    std::vector<std::string> org_ext_id;
    std::vector<std::string> location_ext_id;
    std::vector<std::string> batch_ext_id;
    std::vector<std::string> pack_ext_id;
};
