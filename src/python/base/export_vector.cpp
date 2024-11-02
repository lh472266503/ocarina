//
// Created by Zero on 2024/11/2.
//

#include "export_vector_func.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector(py::module &m) {

#define OC_EXPORT_VECTOR(T, N) [&] {                                                                             \
    py::class_<ocarina::detail::VectorStorage<T, N>>(m, "_VectorStorage" #T #N);                                 \
    auto m##T = export_type<Vector<T, N>, ocarina::detail::VectorStorage<T, N>>(m, #T #N)                        \
                    .def(py::init<T>())                                                                          \
                    .def(py::init<T, T>())                                                                       \
                    .def("__repr__", [](Vector<T, N> &self) { return format(#T #N "({},{})", self.x, self.y); }) \
                    .def("__getitem__", [](Vector<T, N> &self, size_t i) { return self[i]; })                    \
                    .def("__setitem__", [](Vector<T, N> &self, size_t i, T k) { self[i] = k; })                  \
                    .def_readwrite("x", &Vector<T, N>::x)                                                        \
                    .def_readwrite("y", &Vector<T, N>::y);                                                       \
    return m##T;                                                                                                 \
}();

    auto r = OC_EXPORT_VECTOR(float, 2);
}