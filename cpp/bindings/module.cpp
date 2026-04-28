#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "py_engine_input_owner.hpp"
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
    const char* field_name
) {
    try {
        nb::object obj = src.attr(field_name);
        nb::ndarray<T, nb::ndim<1>, nb::c_contig> a = nb::cast<nb::ndarray<T, nb::ndim<1>, nb::c_contig>>(obj);
        const size_t n = (size_t) a.shape(0);
        std::vector<T> out(n);
        std::memcpy(out.data(), a.data(), n * sizeof(T));
        return out;
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string(field_name) + " expectetd 1-D C-contiguous numpy array of correct dtype");
    }
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

    in.org_type = load_numpy_1d<std::uint8_t>(src, "org_type");
    in.location_org_id = load_numpy_1d<std::uint32_t>(src, "location_org_id");
    in.location_market_id = load_numpy_1d<std::uint32_t>(src, "location_market_id");
    in.location_out_edge_offset = load_numpy_1d<std::uint32_t>(
        src, "location_out_edge_offset");
    in.location_out_edge_id = load_numpy_1d<std::uint32_t>(src, "location_out_edge_id");
    in.edge_src_location_id = load_numpy_1d<std::uint32_t>(src, "edge_src_location_id");
    in.edge_dst_location_id = load_numpy_1d<std::uint32_t>(src, "edge_dst_location_id");
    in.edge_cost = load_numpy_1d<float>(src, "edge_cost");
    in.edge_capacity = load_numpy_1d<std::uint32_t>(src, "edge_capacity");
    in.batch_product_id = load_numpy_1d<std::uint32_t>(src, "batch_product_id");
    in.batch_manufacturer_org_id = load_numpy_1d<std::uint32_t>(src, "batch_manufacturer_org_id");
    in.batch_intended_market_offset = load_numpy_1d<std::uint32_t>(src, "batch_intended_market_offset");
    in.batch_intended_market_id = load_numpy_1d<std::uint32_t>(src, "batch_intended_market_id");
    in.pack_product_id = load_numpy_1d<std::uint32_t>(src, "pack_product_id");
    in.pack_batch_id = load_numpy_1d<std::uint32_t>(src, "pack_batch_id");
    in.pack_initial_location_id = load_numpy_1d<std::uint32_t>(src, "pack_initial_location_id");
    in.pack_initial_market_id = load_numpy_1d<std::uint32_t>(src, "pack_initial_market_id");
    in.pack_initial_state = load_numpy_1d<std::uint8_t>(src, "pack_initial_state");

    in.pack_serial = nb::cast<std::vector<std::string>>(src.attr("pack_serial"));
    in.location_has_behavior = load_numpy_1d<std::uint8_t>(src, "location_has_behavior");
    in.location_verify_prob = load_numpy_1d<float>(src, "location_verify_prob");
    in.location_decommission_prob = load_numpy_1d<float>(src, "location_decommission_prob");
    in.location_reactivate_prob = load_numpy_1d<float>(src, "location_reactivate_prob");

    in.org_ext_id = nb::cast<std::vector<std::string>>(src.attr("org_ext_id"));
    in.location_ext_id = nb::cast<std::vector<std::string>>(src.attr("location_ext_id"));
    in.batch_ext_id = nb::cast<std::vector<std::string>>(src.attr("batch_ext_id"));
    in.pack_ext_id = nb::cast<std::vector<std::string>>(src.attr("pack_ext_id"));

    return in;
}

static Simulator create_simulator(const nb::object& py_in) {
    auto owner = build_py_engine_input_owner(py_in);
    std::shared_ptr<const void> opaque = owner;
    return Simulator(std::move(opaque), owner->view);
}

NB_MODULE(_pharmasim_native, m) {
    m.def("create_simulator", &create_simulator);

    nb::class_<Simulator>(m, "Simulator")
        .def("run_ticks", &Simulator::run_ticks)
        .def("current_tick", &Simulator::current_tick)
        .def("event_count", &Simulator::event_count)
        .def("registry_matches_physical", &Simulator::registry_matches_physical)
        .def(
            "event_log_ticks",
            [](const Simulator& s) { return s.events().tick; }
        )
        .def(
            "event_log_pack_ids",
            [](const Simulator& s) { return s.events().pack_id; }
        )
        .def(
            "event_log_types",
            [](const Simulator& s) { return s.events().event_type; }
        )
        .def(
            "event_log_from_locations",
            [](const Simulator& s) { return s.events().from_location_id; }
        )
        .def(
            "event_log_to_locations",
            [](const Simulator& s) { return s.events().to_location_id; }
        )
        .def(
            "physical_pack_states",
            [](const Simulator& s) { return s.state().pack_state; }
        )
        .def(
            "physical_pack_location_ids",
            [](const Simulator& s) { return s.state().pack_location_id; }
        )
        .def(
            "physical_pack_market_ids",
            [](const Simulator& s) { return s.state().pack_market_id; }
        );
}