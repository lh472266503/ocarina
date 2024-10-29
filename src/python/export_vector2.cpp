//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_VECTOR2(type)                                                                       \
    py::class_<ocarina::detail::VectorStorage<type, 2>>(m, "_VectorStorage" #type "2");               \
    auto m##type = py::class_<Vector<type, 2>, ocarina::detail::VectorStorage<type, 2>>(m, #type "2") \
                       .def(py::init<>())                                                             \
                       .def(py::init<type>())                                                         \
                       .def(py::init<type, type>())                                                   \
                       .def("__repr__", [](Vector<type, 2> &self) { return format(#type "2({},{})", self.x, self.y); })

void export_vector2(py::module &m) {
    OC_EXPORT_VECTOR2(float);
}