#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include "fcnn.h"

namespace py = pybind11;

PYBIND11_MODULE(hyperpy, m) {
    xt::import_numpy();
    m.doc() = "The hyper module";

    py::class_<hyper::FCNN>(m, "FCNN")
        .def(py::init<const xt::pyarray<std::size_t>&>())
        .def("forward", [](hyper::FCNN& self, const xt::pyarray<double>& input) {
            return self.forward(input);
        }, py::arg("input"))
        .def_property_readonly("weights", &hyper::FCNN::get_weights);
}