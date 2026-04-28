#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "engine_input_view.hpp"
#include "engine_input_validate.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

struct PyEngineInputOwner {
    nb::object py_engine_input;  // increments ref count so corresponding dataclass stays alive

    nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig> org_type;

    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> location_org_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> location_market_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> location_out_edge_offset;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> location_out_edge_id;

    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> edge_src_location_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> edge_dst_location_id;
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> edge_cost;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> edge_capacity;

    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> batch_product_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> batch_manufacturer_org_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> batch_intended_market_offset;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> batch_intended_market_id;

    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> pack_product_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> pack_batch_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> pack_initial_location_id;
    nb::ndarray<std::uint32_t, nb::ndim<1>, nb::c_contig> pack_initial_market_id;
    nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig> pack_initial_state;

    nb::ndarray<std::uint8_t, nb::ndim<1>, nb::c_contig> location_has_behavior;
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> location_verify_prob;
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> location_decommission_prob;
    nb::ndarray<float, nb::ndim<1>, nb::c_contig> location_reactivate_prob;

    EngineInputView view;
};

template <typename T>
static nb::ndarray<T, nb::ndim<1>, nb::c_contig> require_1d_contig(const nb::object& src, const char* name) {
    try {
        return nb::cast<nb::ndarray<T, nb::ndim<1>, nb::c_contig>>(src.attr(name));   
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string(name) + " expected 1-D C-contiguous numpy array of correct dtype");
    }
}

template <typename T>
static std::span<const T> as_span(const nb::ndarray<T, nb::ndim<1>, nb::c_contig>& arr) {
    return std::span<const T>(arr.data(), (std::size_t) arr.shape(0));
}

inline std::shared_ptr<PyEngineInputOwner> build_py_engine_input_owner(const nb::object& py_in) {
    auto owner = std::make_shared<PyEngineInputOwner>();
    owner->py_engine_input = py_in;
    // metadata
    owner->view.schema_version   = nb::cast<std::string>(py_in.attr("schema_version"));
    owner->view.seed             = nb::cast<std::int64_t>(py_in.attr("seed"));
    owner->view.n_organizations  = nb::cast<int>(py_in.attr("n_organizations"));
    owner->view.n_locations      = nb::cast<int>(py_in.attr("n_locations"));
    owner->view.n_products       = nb::cast<int>(py_in.attr("n_products"));
    owner->view.n_batches        = nb::cast<int>(py_in.attr("n_batches"));
    owner->view.n_packs          = nb::cast<int>(py_in.attr("n_packs"));
    owner->view.n_markets        = nb::cast<int>(py_in.attr("n_markets"));
    owner->view.n_edges          = nb::cast<int>(py_in.attr("n_edges"));
    // numeric ndarrays
    owner->org_type = require_1d_contig<std::uint8_t>(py_in, "org_type");
    owner->view.org_type = as_span(owner->org_type);
    owner->location_org_id = require_1d_contig<std::uint32_t>(py_in, "location_org_id");
    owner->view.location_org_id = as_span(owner->location_org_id);
    owner->location_market_id = require_1d_contig<std::uint32_t>(py_in, "location_market_id");
    owner->view.location_market_id = as_span(owner->location_market_id);
    owner->location_out_edge_offset = require_1d_contig<std::uint32_t>(py_in, "location_out_edge_offset");
    owner->view.location_out_edge_offset = as_span(owner->location_out_edge_offset);
    owner->location_out_edge_id = require_1d_contig<std::uint32_t>(py_in, "location_out_edge_id");
    owner->view.location_out_edge_id = as_span(owner->location_out_edge_id);
    owner->edge_src_location_id = require_1d_contig<std::uint32_t>(py_in, "edge_src_location_id");
    owner->view.edge_src_location_id = as_span(owner->edge_src_location_id);
    owner->edge_dst_location_id = require_1d_contig<std::uint32_t>(py_in, "edge_dst_location_id");
    owner->view.edge_dst_location_id = as_span(owner->edge_dst_location_id);
    owner->edge_cost = require_1d_contig<float>(py_in, "edge_cost");
    owner->view.edge_cost = as_span(owner->edge_cost);
    owner->edge_capacity = require_1d_contig<std::uint32_t>(py_in, "edge_capacity");
    owner->view.edge_capacity = as_span(owner->edge_capacity);
    owner->batch_product_id = require_1d_contig<std::uint32_t>(py_in, "batch_product_id");
    owner->view.batch_product_id = as_span(owner->batch_product_id);
    owner->batch_manufacturer_org_id = require_1d_contig<std::uint32_t>(py_in, "batch_manufacturer_org_id");
    owner->view.batch_manufacturer_org_id = as_span(owner->batch_manufacturer_org_id);
    owner->batch_intended_market_offset = require_1d_contig<std::uint32_t>(py_in, "batch_intended_market_offset");
    owner->view.batch_intended_market_offset = as_span(owner->batch_intended_market_offset);
    owner->batch_intended_market_id = require_1d_contig<std::uint32_t>(py_in, "batch_intended_market_id");
    owner->view.batch_intended_market_id = as_span(owner->batch_intended_market_id);
    owner->pack_product_id = require_1d_contig<std::uint32_t>(py_in, "pack_product_id");
    owner->view.pack_product_id = as_span(owner->pack_product_id);
    owner->pack_batch_id = require_1d_contig<std::uint32_t>(py_in, "pack_batch_id");
    owner->view.pack_batch_id = as_span(owner->pack_batch_id);
    owner->pack_initial_location_id = require_1d_contig<std::uint32_t>(py_in, "pack_initial_location_id");
    owner->view.pack_initial_location_id = as_span(owner->pack_initial_location_id);
    owner->pack_initial_market_id = require_1d_contig<std::uint32_t>(py_in, "pack_initial_market_id");
    owner->view.pack_initial_market_id = as_span(owner->pack_initial_market_id);
    owner->pack_initial_state = require_1d_contig<std::uint8_t>(py_in, "pack_initial_state");
    owner->view.pack_initial_state = as_span(owner->pack_initial_state);
    owner->location_has_behavior = require_1d_contig<std::uint8_t>(py_in, "location_has_behavior");
    owner->view.location_has_behavior = as_span(owner->location_has_behavior);
    owner->location_verify_prob = require_1d_contig<float>(py_in, "location_verify_prob");
    owner->view.location_verify_prob = as_span(owner->location_verify_prob);
    owner->location_decommission_prob = require_1d_contig<float>(py_in, "location_decommission_prob");
    owner->view.location_decommission_prob = as_span(owner->location_decommission_prob);
    owner->location_reactivate_prob = require_1d_contig<float>(py_in, "location_reactivate_prob");
    owner->view.location_reactivate_prob = as_span(owner->location_reactivate_prob);
    // strings
    owner->view.market_code = nb::cast<std::vector<std::string>>(py_in.attr("market_code"));
    owner->view.pack_serial = nb::cast<std::vector<std::string>>(py_in.attr("pack_serial"));
    owner->view.org_ext_id = nb::cast<std::vector<std::string>>(py_in.attr("org_ext_id"));
    owner->view.location_ext_id = nb::cast<std::vector<std::string>>(py_in.attr("location_ext_id"));
    owner->view.batch_ext_id = nb::cast<std::vector<std::string>>(py_in.attr("batch_ext_id"));
    owner->view.pack_ext_id = nb::cast<std::vector<std::string>>(py_in.attr("pack_ext_id"));
    // structural validation
    validate_engine_input_or_throw(owner->view);
    return owner;
}

