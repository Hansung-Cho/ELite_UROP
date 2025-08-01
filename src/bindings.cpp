#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "height_filter.h"

namespace py = pybind11;

PYBIND11_MODULE(elite_utils, m) {
    m.doc() = "ELite C++ 유틸 바인딩";

    py::class_<Point>(m, "Point")
        .def(py::init<double, double, double>(),
            py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("z", &Point::z);

    m.def("height_filter",
          &height_filter,
          "Filter points by z range",
          py::arg("pts"),
          py::arg("min_z"));
}
