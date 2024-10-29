//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"
#include "swizzle_inl/swizzle3.inl.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_VECTOR3(T)                                                                    \
    py::class_<ocarina::detail::VectorStorage<T, 3>>(m, "_VectorStorage" #T "3");               \
    auto m##T = py::class_<Vector<T, 3>, ocarina::detail::VectorStorage<T, 3>>(m, #T "3")       \
                    .def(py::init<>())                                                          \
                    .def(py::init<T>())                                                         \
                    .def(py::init<T, T, T>())                                                   \
                    .def("__repr__", [](Vector<T, 3> &self) {                                   \
                        return format(#T "3({},{},{})", self.x, self.y, self.z);                \
                    })                                                                          \
                    .def("__getitem__", [](Vector<T, 3> &self, size_t i) { return self[i]; })   \
                    .def("__setitem__", [](Vector<T, 3> &self, size_t i, T k) { self[i] = k; }) \
                    .def("copy", [](Vector<T, 3> &self) { return Vector<T, 3>(self); })         \
                    .def_readwrite("x", &Vector<T, 3>::x)                                       \
                    .def_readwrite("y", &Vector<T, 3>::y)                                       \
                    .def_readwrite("z", &Vector<T, 3>::z);                                      \
    export_swizzle3<T>(m##T);

void export_vector3(py::module &m) {
    OC_EXPORT_VECTOR3(float)
    OC_EXPORT_VECTOR3(uint);
    OC_EXPORT_VECTOR3(int);
    OC_EXPORT_VECTOR3(bool);
}