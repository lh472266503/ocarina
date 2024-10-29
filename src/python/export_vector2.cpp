//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector_func.h"
#include "swizzle_inl/swizzle2.inl.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_MAKE_VECTOR2(T)                                      \
    m.def("make_" #T "2", [](T a) { return make_##T##2(a); });         \
    m.def("make_" #T "2", [](T a, T b) { return make_##T##2(a, b); }); \
    m.def("make_" #T "2", [](Vector<T, 2> a) { return make_##T##2(a); });

#define OC_EXPORT_VECTOR2(T)                                                                                   \
    py::class_<ocarina::detail::VectorStorage<T, 2>>(m, "_VectorStorage" #T "2");                              \
    auto m##T = py::class_<Vector<T, 2>, ocarina::detail::VectorStorage<T, 2>>(m, #T "2")                      \
                    .def(py::init<>())                                                                         \
                    .def(py::init<T>())                                                                        \
                    .def(py::init<T, T>())                                                                     \
                    .def("__repr__", [](Vector<T, 2> &self) { return format(#T "2({},{})", self.x, self.y); }) \
                    .def("__getitem__", [](Vector<T, 2> &self, size_t i) { return self[i]; })                  \
                    .def("__setitem__", [](Vector<T, 2> &self, size_t i, T k) { self[i] = k; })                \
                    .def("copy", [](Vector<T, 2> &self) { return Vector<T, 2>(self); })                        \
                    .def_readwrite("x", &Vector<T, 2>::x)                                                      \
                    .def_readwrite("y", &Vector<T, 2>::y);                                                     \
    export_swizzle2<T>(m##T);                                                                                  \
    export_all_func<T, 2>(m##T);                                                                              \
    OC_EXPORT_MAKE_VECTOR2(T)

void export_vector2(py::module &m) {
    OC_EXPORT_VECTOR2(float);
    OC_EXPORT_VECTOR2(uint);
    OC_EXPORT_VECTOR2(int);
    OC_EXPORT_VECTOR2(bool);
}