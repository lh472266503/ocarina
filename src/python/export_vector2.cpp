//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

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
                    .def_readwrite("y", &Vector<T, 2>::y)                                                      \
                    .def_readwrite("xx", [](Vector<T, 2> &self) { return self.xx().decay(); })   \
                    .def_readwrite("xy", [](Vector<T, 2> &self) { return self.xy().decay(); })

void export_vector2(py::module &m) {
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
                    .def_readwrite("y", &Vector<T, 2>::y)                                                      \
                    .def_readwrite("xx", [](Vector<T, 2> &self) { return self.xx().decay(); })   \
                    .def_readwrite("xy", [](Vector<T, 2> &self) { return self.xy().decay(); })
Replacement:
    py::class_<ocarina::detail::VectorStorage<float, 2>>(m, "_VectorStorage"
                                                            "float"
                                                            "2");
    auto mfloat = py::class_<Vector<float, 2>, ocarina::detail::VectorStorage<float, 2>>(m, "float"
                                                                                            "2")
                      .def(py::init<>())
                      .def(py::init<float>())
                      .def(py::init<float, float>())
                      .def("__repr__", [](Vector<float, 2> &self) { return format("float"
                                                                                  "2({},{})",
                                                                                  self.x, self.y); })
                      .def("__getitem__", [](Vector<float, 2> &self, size_t i) { return self[i]; })
                      .def("__setitem__", [](Vector<float, 2> &self, size_t i, float k) { self[i] = k; })
                      .def("copy", [](Vector<float, 2> &self) { return Vector<float, 2>(self); })
                      .def_readwrite("x", &Vector<float, 2>::x)
                      .def_readwrite("y", &Vector<float, 2>::y)
                      .def_property("xx", [](Vector<float, 2> &self) { return self.xx().decay(); }, [](Vector<float, 2> &self, Vector<float, 2> v) {
                              self.xx() = v;
                          });
//    OC_EXPORT_VECTOR2(float);
}