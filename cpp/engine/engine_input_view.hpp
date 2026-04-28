#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <span>

struct EngineInputView {
    // metadata
    std::string schema_version;
    std::int64_t seed;
    int n_organizations{}, n_locations{}, n_products{}, n_batches{}, n_packs{}, n_markets{}, n_edges{};

    // numeric columns: non-owning
    std::span<const std::uint8_t> org_type;

    std::span<const std::uint32_t> location_org_id;
    std::span<const std::uint32_t> location_market_id;
    std::span<const std::uint32_t> location_out_edge_offset;
    std::span<const std::uint32_t> location_out_edge_id;

    std::span<const std::uint32_t> edge_src_location_id;
    std::span<const std::uint32_t> edge_dst_location_id;
    std::span<const float> edge_cost;
    std::span<const std::uint32_t> edge_capacity;

    std::span<const std::uint32_t> batch_product_id;
    std::span<const std::uint32_t> batch_manufacturer_org_id;
    std::span<const std::uint32_t> batch_intended_market_offset;
    std::span<const std::uint32_t> batch_intended_market_id;

    std::span<const std::uint32_t> pack_product_id;
    std::span<const std::uint32_t> pack_batch_id;
    std::span<const std::uint32_t> pack_initial_location_id;
    std::span<const std::uint32_t> pack_initial_market_id;
    std::span<const std::uint8_t> pack_initial_state;

    std::span<const std::uint8_t> location_has_behavior;
    std::span<const float> location_verify_prob;
    std::span<const float> location_decommission_prob;
    std::span<const float> location_reactivate_prob;

    // strings: must own, don't want to deal with no-copy moving between python strings and cpp std::string
    std::vector<std::string> market_code;
    std::vector<std::string> pack_serial;
    std::vector<std::string> org_ext_id, location_ext_id, batch_ext_id, pack_ext_id;
};