#pragma once

#include "engine_input.hpp"
#include "engine_input_view.hpp"

#include <span>
#include <cstddef>

inline EngineInputView make_view(const EngineInput& in) {
    EngineInputView v{};

    // metadata
    v.schema_version    = in.schema_version;
    v.seed              = in.seed;
    v.n_organizations   = in.n_organizations;
    v.n_locations       = in.n_locations;
    v.n_products        = in.n_products;
    v.n_batches         = in.n_batches;
    v.n_packs           = in.n_packs;
    v.n_markets         = in.n_markets;
    v.n_edges           = in.n_edges;

    // numeric spans: point into std::vector
    v.org_type = std::span<const std::uint8_t>(in.org_type.data(), in.org_type.size());
    
    v.location_org_id = std::span<const std::uint32_t>(in.location_org_id.data(), in.location_org_id.size());
    v.location_market_id = std::span<const std::uint32_t>(in.location_market_id.data(), in.location_market_id.size());
    v.location_out_edge_offset = std::span<const std::uint32_t>(in.location_out_edge_offset.data(), in.location_out_edge_offset.size());
    v.location_out_edge_id = std::span<const std::uint32_t>(in.location_out_edge_id.data(), in.location_out_edge_id.size());
    
    v.edge_src_location_id = std::span<const std::uint32_t>(in.edge_src_location_id.data(), in.edge_src_location_id.size());
    v.edge_dst_location_id = std::span<const std::uint32_t>(in.edge_dst_location_id.data(), in.edge_dst_location_id.size());
    v.edge_cost = std::span<const float>(in.edge_cost.data(), in.edge_cost.size());
    v.edge_capacity = std::span<const std::uint32_t>(in.edge_capacity.data(), in.edge_capacity.size());
    
    v.batch_product_id = std::span<const std::uint32_t>(in.batch_product_id.data(), in.batch_product_id.size());
    v.batch_manufacturer_org_id = std::span<const std::uint32_t>(in.batch_manufacturer_org_id.data(), in.batch_manufacturer_org_id.size());
    v.batch_intended_market_offset = std::span<const std::uint32_t>(in.batch_intended_market_offset.data(), in.batch_intended_market_offset.size());
    v.batch_intended_market_id = std::span<const std::uint32_t>(in.batch_intended_market_id.data(), in.batch_intended_market_id.size());

    v.pack_product_id = std::span<const std::uint32_t>(in.pack_product_id.data(), in.pack_product_id.size());
    v.pack_batch_id = std::span<const std::uint32_t>(in.pack_batch_id.data(), in.pack_batch_id.size());
    v.pack_initial_location_id = std::span<const std::uint32_t>(in.pack_initial_location_id.data(), in.pack_initial_location_id.size());
    v.pack_initial_market_id = std::span<const std::uint32_t>(in.pack_initial_market_id.data(), in.pack_initial_market_id.size());
    v.pack_initial_state = std::span<const std::uint8_t>(in.pack_initial_state.data(), in.pack_initial_state.size());

    v.location_has_behavior = std::span<const std::uint8_t>(in.location_has_behavior.data(), in.location_has_behavior.size());
    v.location_verify_prob = std::span<const float>(in.location_verify_prob.data(), in.location_verify_prob.size());
    v.location_decommission_prob = std::span<const float>(in.location_decommission_prob.data(), in.location_decommission_prob.size());
    v.location_reactivate_prob = std::span<const float>(in.location_reactivate_prob.data(), in.location_reactivate_prob.size());

    v.location_demand_policy_id = std::span<const std::uint8_t>(in.location_demand_policy_id.data(), in.location_demand_policy_id.size());
    v.location_supply_policy_id = std::span<const std::uint8_t>(in.location_supply_policy_id.data(), in.location_supply_policy_id.size());
    v.location_initial_on_hand = std::span<const std::int32_t>(in.location_initial_on_hand.data(), in.location_initial_on_hand.size());
    v.location_initial_backlog = std::span<const std::int32_t>(in.location_initial_backlog.data(), in.location_initial_backlog.size());
    v.location_initial_pipeline_outstanding = std::span<const std::int32_t>(in.location_initial_pipeline_outstanding.data(), in.location_initial_pipeline_outstanding.size());
    
    v.location_demand_const_rate = std::span<const std::int32_t>(in.location_demand_const_rate.data(), in.location_demand_const_rate.size());
    v.location_demand_poisson_lambda = std::span<const float>(in.location_demand_poisson_lambda.data(), in.location_demand_poisson_lambda.size());
    v.location_reorder_point_s = std::span<const std::int32_t>(in.location_reorder_point_s.data(), in.location_reorder_point_s.size());
    v.location_order_up_to_S = std::span<const std::int32_t>(in.location_order_up_to_S.data(), in.location_order_up_to_S.size());
    v.location_base_stock_level = std::span<const std::int32_t>(in.location_base_stock_level.data(), in.location_base_stock_level.size());
    v.location_ewma_alpha = std::span<const float>(in.location_ewma_alpha.data(), in.location_ewma_alpha.size());
    v.location_supply_capacity_per_tick = std::span<const std::uint32_t>(in.location_supply_capacity_per_tick.data(), in.location_supply_capacity_per_tick.size());
    v.location_min_order_interval_ticks = std::span<const std::uint32_t>(in.location_min_order_interval_ticks.data(), in.location_min_order_interval_ticks.size());
    v.location_unfulfilled_unit_penalty = std::span<const float>(in.location_unfulfilled_unit_penalty.data(), in.location_unfulfilled_unit_penalty.size());
    v.location_preferred_supplier_edge_id = std::span<const std::uint32_t>(in.location_preferred_supplier_edge_id.data(), in.location_preferred_supplier_edge_id.size());
    
    v.edge_lead_time_ticks = std::span<const std::uint16_t>(in.edge_lead_time_ticks.data(), in.edge_lead_time_ticks.size());

    v.location_route_offset = std::span<const std::uint32_t>(in.location_route_offset.data(), in.location_route_offset.size());
    v.location_route_dst_location_id = std::span<const std::uint32_t>(in.location_route_dst_location_id.data(), in.location_route_dst_location_id.size());
    v.location_route_next_edge_id = std::span<const std::uint32_t>(in.location_route_next_edge_id.data(), in.location_route_next_edge_id.size());

    v.market_code = in.market_code;
    v.pack_serial = in.pack_serial;
    v.org_ext_id = in.org_ext_id;
    v.location_ext_id = in.location_ext_id;
    v.batch_ext_id = in.batch_ext_id;
    v.pack_ext_id = in.pack_ext_id;

    return v;
}