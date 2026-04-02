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

    std::vector<std::string> org_ext_id;
    std::vector<std::string> location_ext_id;
    std::vector<std::string> batch_ext_id;
    std::vector<std::string> pack_ext_id;
};
