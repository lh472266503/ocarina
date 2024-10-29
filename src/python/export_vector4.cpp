//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"
#include "swizzle_inl/swizzle4.inl.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_MAKE_VECTOR4(T)                                                      \
    m.def("make_" #T "4", [](T a) { return make_##T##4(a); });                         \
    m.def("make_" #T "4", [](T a, T b, T c, T d) { return make_##T##4(a, b, c, d); }); \
    m.def("make_" #T "4", [](Vector<T, 4> a) { return make_##T##4(a); });


#define OC_EXPORT_VECTOR4(T)                                                                    \
    py::class_<ocarina::detail::VectorStorage<T, 4>>(m, "_VectorStorage" #T "4");               \
    auto m##T = py::class_<Vector<T, 4>, ocarina::detail::VectorStorage<T, 4>>(m, #T "4")       \
                    .def(py::init<>())                                                          \
                    .def(py::init<T>())                                                         \
                    .def(py::init<T, T, T, T>())                                                \
                    .def("__repr__", [](Vector<T, 4> &self) {                                   \
                        return format(#T "4({},{},{},{})", self.x, self.y, self.z, self.w);     \
                    })                                                                          \
                    .def("__getitem__", [](Vector<T, 4> &self, size_t i) { return self[i]; })   \
                    .def("__setitem__", [](Vector<T, 4> &self, size_t i, T k) { self[i] = k; }) \
                    .def("copy", [](Vector<T, 4> &self) { return Vector<T, 4>(self); })         \
                    .def_readwrite("x", &Vector<T, 4>::x)                                       \
                    .def_readwrite("y", &Vector<T, 4>::y)                                       \
                    .def_readwrite("z", &Vector<T, 4>::z)                                       \
                    .def_readwrite("w", &Vector<T, 4>::w);                                      \
    export_swizzle4<T>(m##T);                                                                   \
    OC_EXPORT_MAKE_VECTOR4(T)

void export_vector4(py::module &m) {
    OC_EXPORT_VECTOR4(float)
    OC_EXPORT_VECTOR4(uint);
    OC_EXPORT_VECTOR4(int);
    OC_EXPORT_VECTOR4(bool);
}