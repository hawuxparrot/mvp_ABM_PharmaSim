#include "engine_input_validate.hpp"

#include "enums.hpp"

#include <stdexcept>

std::string validate_engine_input_message(const EngineInput& in) noexcept {
    try {
        validate_engine_input_or_throw(in);
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return {};
}

void validate_engine_input_or_throw(const EngineInput& in) {
    auto fail = [](const char* msg) { throw std::invalid_argument(msg); };

    if (in.schema_version != ENGINE_INPUT_SCHEMA_VERSION) {
        fail("EngineInput.schema_version must match ENGINE_INPUT_SCHEMA_VERSION (engine_input.v3)");
    }

    const auto n_org = static_cast<std::size_t>(in.n_organizations);
    const auto n_loc = static_cast<std::size_t>(in.n_locations);
    const auto n_batch = static_cast<std::size_t>(in.n_batches);
    const auto n_pack = static_cast<std::size_t>(in.n_packs);
    const auto n_mkt = static_cast<std::size_t>(in.n_markets);
    const auto n_edge = static_cast<std::size_t>(in.n_edges);

    if (in.n_organizations < 0 || in.n_locations < 0 || in.n_products < 0 || in.n_batches < 0
        || in.n_packs < 0 || in.n_markets < 0 || in.n_edges < 0) {
        fail("EngineInput n_* counts must be non-negative");
    }

    if (in.market_code.size() != n_mkt) {
        fail("EngineInput.market_code length must match n_markets");
    }
    if (in.org_type.size() != n_org) {
        fail("EngineInput.org_type length must match n_organizations");
    }
    if (in.location_org_id.size() != n_loc || in.location_market_id.size() != n_loc) {
        fail("EngineInput location columns length must match n_locations");
    }
    if (in.location_out_edge_offset.size() != n_loc + 1) {
        fail("EngineInput.location_out_edge_offset length must be n_locations + 1");
    }
    if (in.location_out_edge_id.size() != n_edge) {
        fail("EngineInput.location_out_edge_id length must match n_edges");
    }
    if (in.edge_src_location_id.size() != n_edge || in.edge_dst_location_id.size() != n_edge
        || in.edge_cost.size() != n_edge || in.edge_capacity.size() != n_edge) {
        fail("EngineInput edge columns length must match n_edges");
    }
    if (in.batch_product_id.size() != n_batch || in.batch_manufacturer_org_id.size() != n_batch) {
        fail("EngineInput batch id columns length must match n_batches");
    }
    if (in.batch_intended_market_offset.size() != n_batch + 1) {
        fail("EngineInput.batch_intended_market_offset length must be n_batches + 1");
    }
    if (in.pack_product_id.size() != n_pack || in.pack_batch_id.size() != n_pack
        || in.pack_initial_location_id.size() != n_pack || in.pack_initial_market_id.size() != n_pack
        || in.pack_initial_state.size() != n_pack || in.pack_serial.size() != n_pack) {
        fail("EngineInput pack columns length must match n_packs");
    }

    if (!in.org_ext_id.empty() && in.org_ext_id.size() != n_org) {
        fail("EngineInput.org_ext_id must be empty or length n_organizations");
    }
    if (!in.location_ext_id.empty() && in.location_ext_id.size() != n_loc) {
        fail("EngineInput.location_ext_id must be empty or length n_locations");
    }
    if (!in.batch_ext_id.empty() && in.batch_ext_id.size() != n_batch) {
        fail("EngineInput.batch_ext_id must be empty or length n_batches");
    }
    if (!in.pack_ext_id.empty() && in.pack_ext_id.size() != n_pack) {
        fail("EngineInput.pack_ext_id must be empty or length n_packs");
    }

    if (in.location_has_behavior.size() != n_loc || in.location_verify_prob.size() != n_loc
        || in.location_decommission_prob.size() != n_loc
        || in.location_reactivate_prob.size() != n_loc) {
        fail("EngineInput location behavior columns must each have length n_locations");
    }

    const auto& off = in.batch_intended_market_offset;
    const auto& flat = in.batch_intended_market_id;
    if (off.empty()) {
        fail("EngineInput.batch_intended_market_offset must not be empty");
    }
    if (off.front() != 0) {
        fail("EngineInput.batch_intended_market_offset[0] must be 0");
    }
    if (off.back() != flat.size()) {
        fail("EngineInput.batch_intended_market_offset.back() must equal batch_intended_market_id.size()");
    }
    for (std::size_t i = 1; i < off.size(); ++i) {
        if (off[i] < off[i - 1]) {
            fail("EngineInput.batch_intended_market_offset must be non-decreasing");
        }
    }

    const auto& edge_off = in.location_out_edge_offset;
    const auto& edge_ids = in.location_out_edge_id;
    if (edge_off.empty()) {
        fail("EngineInput.location_out_edge_offset must not be empty");
    }
    if (edge_off.front() != 0) {
        fail("EngineInput.location_out_edge_offset[0] must be 0");
    }
    if (edge_off.back() != edge_ids.size()) {
        fail("EngineInput.location_out_edge_offset.back() must equal location_out_edge_id.size()");
    }
    for (std::size_t i = 1; i < edge_off.size(); ++i) {
        if (edge_off[i] < edge_off[i - 1]) {
            fail("EngineInput.location_out_edge_offset must be non-decreasing");
        }
    }
    for (std::size_t i = 0; i < edge_ids.size(); ++i) {
        const auto edge_id = static_cast<std::size_t>(edge_ids[i]);
        if (edge_id >= n_edge) {
            fail("EngineInput.location_out_edge_id entries must be < n_edges");
        }
    }
    for (std::size_t i = 0; i < n_edge; ++i) {
        const auto src = static_cast<std::size_t>(in.edge_src_location_id[i]);
        const auto dst = static_cast<std::size_t>(in.edge_dst_location_id[i]);
        if (src >= n_loc || dst >= n_loc) {
            fail("EngineInput edge src/dst ids must be < n_locations");
        }
    }
    for (std::size_t loc = 0; loc < n_loc; ++loc) {
        const auto start = static_cast<std::size_t>(edge_off[loc]);
        const auto end = static_cast<std::size_t>(edge_off[loc + 1]);
        for (std::size_t k = start; k < end; ++k) {
            const auto edge_id = static_cast<std::size_t>(edge_ids[k]);
            const auto src = static_cast<std::size_t>(in.edge_src_location_id[edge_id]);
            if (src != loc) {
                fail("EngineInput location_out_edge CSR must group edges by matching source location");
            }
        }
    }
}

void validate_engine_input_or_throw(const EngineInputView& in) {
    auto fail = [](const char* msg) { throw std::invalid_argument(msg);};

    if (in.schema_version != ENGINE_INPUT_SCHEMA_VERSION) {
        fail("EngineInput.schema_version must match ENGINE_INPUT_SCHEMA_VERSION (engine_input.v3)");
    }

    const auto n_org = static_cast<std::size_t>(in.n_organizations);
    const auto n_loc = static_cast<std::size_t>(in.n_locations);
    const auto n_batch = static_cast<std::size_t>(in.n_batches);
    const auto n_pack = static_cast<std::size_t>(in.n_packs);
    const auto n_mkt = static_cast<std::size_t>(in.n_markets);
    const auto n_edge = static_cast<std::size_t>(in.n_edges);

    if (in.n_organizations < 0 || in.n_locations < 0 || in.n_products < 0 || in.n_batches < 0
        || in.n_packs < 0 || in.n_markets < 0 || in.n_edges < 0) {
        fail("EngineInput n_* counts must be non-negative");
    }

    if (in.market_code.size() != n_mkt) {
        fail("EngineInput.market_code length must match n_markets");
    }
    if (in.org_type.size() != n_org) {
        fail("EngineInput.org_type length must match n_organizations");
    }
    if (in.location_org_id.size() != n_loc || in.location_market_id.size() != n_loc) {
        fail("EngineInput location columns length must match n_locations");
    }
    if (in.location_out_edge_offset.size() != n_loc + 1) {
        fail("EngineInput.location_out_edge_offset length must be n_locations + 1");
    }
    if (in.location_out_edge_id.size() != n_edge) {
        fail("EngineInput.location_out_edge_id length must match n_edges");
    }
    if (in.edge_src_location_id.size() != n_edge || in.edge_dst_location_id.size() != n_edge
        || in.edge_cost.size() != n_edge || in.edge_capacity.size() != n_edge) {
        fail("EngineInput edge columns length must match n_edges");
    }
    if (in.batch_product_id.size() != n_batch || in.batch_manufacturer_org_id.size() != n_batch) {
        fail("EngineInput batch id columns length must match n_batches");
    }
    if (in.batch_intended_market_offset.size() != n_batch + 1) {
        fail("EngineInput.batch_intended_market_offset length must be n_batches + 1");
    }
    if (in.pack_product_id.size() != n_pack || in.pack_batch_id.size() != n_pack
        || in.pack_initial_location_id.size() != n_pack || in.pack_initial_market_id.size() != n_pack
        || in.pack_initial_state.size() != n_pack || in.pack_serial.size() != n_pack) {
        fail("EngineInput pack columns length must match n_packs");
    }

    if (!in.org_ext_id.empty() && in.org_ext_id.size() != n_org) {
        fail("EngineInput.org_ext_id must be empty or length n_organizations");
    }
    if (!in.location_ext_id.empty() && in.location_ext_id.size() != n_loc) {
        fail("EngineInput.location_ext_id must be empty or length n_locations");
    }
    if (!in.batch_ext_id.empty() && in.batch_ext_id.size() != n_batch) {
        fail("EngineInput.batch_ext_id must be empty or length n_batches");
    }
    if (!in.pack_ext_id.empty() && in.pack_ext_id.size() != n_pack) {
        fail("EngineInput.pack_ext_id must be empty or length n_packs");
    }

    if (in.location_has_behavior.size() != n_loc || in.location_verify_prob.size() != n_loc
        || in.location_decommission_prob.size() != n_loc
        || in.location_reactivate_prob.size() != n_loc) {
        fail("EngineInput location behavior columns must each have length n_locations");
    }

    const auto& off = in.batch_intended_market_offset;
    const auto& flat = in.batch_intended_market_id;
    if (off.empty()) {
        fail("EngineInput.batch_intended_market_offset must not be empty");
    }
    if (off.front() != 0) {
        fail("EngineInput.batch_intended_market_offset[0] must be 0");
    }
    if (off.back() != flat.size()) {
        fail("EngineInput.batch_intended_market_offset.back() must equal batch_intended_market_id.size()");
    }
    for (std::size_t i = 1; i < off.size(); ++i) {
        if (off[i] < off[i - 1]) {
            fail("EngineInput.batch_intended_market_offset must be non-decreasing");
        }
    }

    const auto& edge_off = in.location_out_edge_offset;
    const auto& edge_ids = in.location_out_edge_id;
    if (edge_off.empty()) {
        fail("EngineInput.location_out_edge_offset must not be empty");
    }
    if (edge_off.front() != 0) {
        fail("EngineInput.location_out_edge_offset[0] must be 0");
    }
    if (edge_off.back() != edge_ids.size()) {
        fail("EngineInput.location_out_edge_offset.back() must equal location_out_edge_id.size()");
    }
    for (std::size_t i = 1; i < edge_off.size(); ++i) {
        if (edge_off[i] < edge_off[i - 1]) {
            fail("EngineInput.location_out_edge_offset must be non-decreasing");
        }
    }
    for (std::size_t i = 0; i < edge_ids.size(); ++i) {
        const auto edge_id = static_cast<std::size_t>(edge_ids[i]);
        if (edge_id >= n_edge) {
            fail("EngineInput.location_out_edge_id entries must be < n_edges");
        }
    }
    for (std::size_t i = 0; i < n_edge; ++i) {
        const auto src = static_cast<std::size_t>(in.edge_src_location_id[i]);
        const auto dst = static_cast<std::size_t>(in.edge_dst_location_id[i]);
        if (src >= n_loc || dst >= n_loc) {
            fail("EngineInput edge src/dst ids must be < n_locations");
        }
    }
    for (std::size_t loc = 0; loc < n_loc; ++loc) {
        const auto start = static_cast<std::size_t>(edge_off[loc]);
        const auto end = static_cast<std::size_t>(edge_off[loc + 1]);
        for (std::size_t k = start; k < end; ++k) {
            const auto edge_id = static_cast<std::size_t>(edge_ids[k]);
            const auto src = static_cast<std::size_t>(in.edge_src_location_id[edge_id]);
            if (src != loc) {
                fail("EngineInput location_out_edge CSR must group edges by matching source location");
            }
        }
    }

}
