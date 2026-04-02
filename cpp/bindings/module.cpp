#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "simulator.hpp"
#include "engine_input.hpp"
#include "engine_input_validate.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

template <typename T>
std::vector<T> load_numpy_1d(
    const nb::object& src,
    const char* field_name,
    const char* expected_dtype_name
) {
    nb::object arr = src.attr(field_name);
    const int ndim = nb::cast<int>(arr.attr("ndim"));
    if (ndim != 1) {
        throw std::invalid_argument(std::string(field_name) + " must be a 1-D numpy array");
    }
    const std::string dtype_name = nb::cast<std::string>(arr.attr("dtype").attr("name"));
    if (dtype_name != expected_dtype_name) {
        throw std::invalid_argument(
            std::string(field_name) + " expected dtype " + expected_dtype_name + ", got "
            + dtype_name
        );
    }
    return nb::cast<std::vector<T>>(arr.attr("tolist")());
}

static EngineInput load_engine_input(const nb::object& src) {
    EngineInput in{};
    in.schema_version = nb::cast<std::string>(src.attr("schema_version"));
    in.seed = nb::cast<std::int64_t>(src.attr("seed"));
    in.n_organizations = nb::cast<int>(src.attr("n_organizations"));
    in.n_locations = nb::cast<int>(src.attr("n_locations"));
    in.n_products = nb::cast<int>(src.attr("n_products"));
    in.n_batches = nb::cast<int>(src.attr("n_batches"));
    in.n_packs = nb::cast<int>(src.attr("n_packs"));
    in.n_markets = nb::cast<int>(src.attr("n_markets"));
    in.n_edges = nb::cast<int>(src.attr("n_edges"));

    in.market_code = nb::cast<std::vector<std::string>>(src.attr("market_code"));

    in.org_type = load_numpy_1d<std::uint8_t>(src, "org_type", "uint8");
    in.location_org_id = load_numpy_1d<std::uint32_t>(src, "location_org_id", "uint32");
    in.location_market_id = load_numpy_1d<std::uint32_t>(src, "location_market_id", "uint32");
    in.location_out_edge_offset = load_numpy_1d<std::uint32_t>(
        src, "location_out_edge_offset", "uint32"
    );
    in.location_out_edge_id = load_numpy_1d<std::uint32_t>(
        src, "location_out_edge_id", "uint32"
    );
    in.edge_src_location_id = load_numpy_1d<std::uint32_t>(
        src, "edge_src_location_id", "uint32"
    );
    in.edge_dst_location_id = load_numpy_1d<std::uint32_t>(
        src, "edge_dst_location_id", "uint32"
    );
    in.edge_cost = load_numpy_1d<float>(src, "edge_cost", "float32");
    in.edge_capacity = load_numpy_1d<std::uint32_t>(src, "edge_capacity", "uint32");
    in.batch_product_id = load_numpy_1d<std::uint32_t>(src, "batch_product_id", "uint32");
    in.batch_manufacturer_org_id = load_numpy_1d<std::uint32_t>(
        src, "batch_manufacturer_org_id", "uint32"
    );
    in.batch_intended_market_offset = load_numpy_1d<std::uint32_t>(
        src, "batch_intended_market_offset", "uint32"
    );
    in.batch_intended_market_id = load_numpy_1d<std::uint32_t>(
        src, "batch_intended_market_id", "uint32"
    );
    in.pack_product_id = load_numpy_1d<std::uint32_t>(src, "pack_product_id", "uint32");
    in.pack_batch_id = load_numpy_1d<std::uint32_t>(src, "pack_batch_id", "uint32");
    in.pack_initial_location_id = load_numpy_1d<std::uint32_t>(
        src, "pack_initial_location_id", "uint32"
    );
    in.pack_initial_market_id = load_numpy_1d<std::uint32_t>(
        src, "pack_initial_market_id", "uint32"
    );
    in.pack_initial_state = load_numpy_1d<std::uint8_t>(src, "pack_initial_state", "uint8");

    in.pack_serial = nb::cast<std::vector<std::string>>(src.attr("pack_serial"));
    in.location_has_behavior = load_numpy_1d<std::uint8_t>(
        src, "location_has_behavior", "uint8"
    );
    in.location_verify_prob = load_numpy_1d<float>(src, "location_verify_prob", "float32");
    in.location_decommission_prob = load_numpy_1d<float>(
        src, "location_decommission_prob", "float32"
    );
    in.location_reactivate_prob = load_numpy_1d<float>(
        src, "location_reactivate_prob", "float32"
    );

    in.org_ext_id = nb::cast<std::vector<std::string>>(src.attr("org_ext_id"));
    in.location_ext_id = nb::cast<std::vector<std::string>>(src.attr("location_ext_id"));
    in.batch_ext_id = nb::cast<std::vector<std::string>>(src.attr("batch_ext_id"));
    in.pack_ext_id = nb::cast<std::vector<std::string>>(src.attr("pack_ext_id"));

    return in;
}

static Simulator create_simulator(const nb::object& py_in) {
    EngineInput in = load_engine_input(py_in);
    validate_engine_input_or_throw(in);
    return Simulator(std::move(in));
}

NB_MODULE(_pharmasim_native, m) {
    m.def("create_simulator", &create_simulator);

    nb::class_<Simulator>(m, "Simulator")
        .def("run_ticks", &Simulator::run_ticks)
        .def("current_tick", &Simulator::current_tick);
}