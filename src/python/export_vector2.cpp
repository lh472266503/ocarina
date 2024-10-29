//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_VECTOR2(T)                                                                                                                         \
    py::class_<ocarina::detail::VectorStorage<T, 2>>(m, "_VectorStorage" #T "2");                                                                    \
    auto m##T = py::class_<Vector<T, 2>, ocarina::detail::VectorStorage<T, 2>>(m, #T "2")                                                            \
                    .def(py::init<>())                                                                                                               \
                    .def(py::init<T>())                                                                                                              \
                    .def(py::init<T, T>())                                                                                                           \
                    .def("__repr__", [](Vector<T, 2> &self) { return format(#T "2({},{})", self.x, self.y); })                                       \
                    .def("__getitem__", [](Vector<T, 2> &self, size_t i) { return self[i]; })                                                        \
                    .def("__setitem__", [](Vector<T, 2> &self, size_t i, T k) { self[i] = k; })                                                      \
                    .def("copy", [](Vector<T, 2> &self) { return Vector<T, 2>(self); })                                                              \
                    .def_readwrite("x", &Vector<float, 2>::x)                                                                                        \
                    .def_readwrite("y", &Vector<float, 2>::y)                                                                                        \
                    .def_property(                                                                                                                   \
                        "xx", [](Vector<T, 2> &self) { return self.xx().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) { self.xx() = v; })       \
                    .def_property(                                                                                                                   \
                        "xy", [](Vector<T, 2> &self) { return self.xy().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) { self.xy() = v; })       \
                    .def_property(                                                                                                                   \
                        "yx", [](Vector<T, 2> &self) { return self.yx().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) { self.yx() = v; })       \
                    .def_property(                                                                                                                   \
                        "yy", [](Vector<T, 2> &self) { return self.yy().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) { self.yy() = v; })       \
                    .def_property(                                                                                                                   \
                        "xxx", [](Vector<T, 2> &self) { return self.xxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.xxx() = v; })    \
                    .def_property(                                                                                                                   \
                        "xxy", [](Vector<T, 2> &self) { return self.xxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.xxy() = v; })    \
                    .def_property(                                                                                                                   \
                        "xyx", [](Vector<T, 2> &self) { return self.xyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.xyx() = v; })    \
                    .def_property(                                                                                                                   \
                        "xyy", [](Vector<T, 2> &self) { return self.xyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.xyy() = v; })    \
                    .def_property(                                                                                                                   \
                        "yxx", [](Vector<T, 2> &self) { return self.yxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.yxx() = v; })    \
                    .def_property(                                                                                                                   \
                        "yxy", [](Vector<T, 2> &self) { return self.yxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.yxy() = v; })    \
                    .def_property(                                                                                                                   \
                        "yyx", [](Vector<T, 2> &self) { return self.yyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.yyx() = v; })    \
                    .def_property(                                                                                                                   \
                        "yyy", [](Vector<T, 2> &self) { return self.yyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) { self.yyy() = v; })    \
                    .def_property(                                                                                                                   \
                        "xxxx", [](Vector<T, 2> &self) { return self.xxxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xxxx() = v; }) \
                    .def_property(                                                                                                                   \
                        "xxxy", [](Vector<T, 2> &self) { return self.xxxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xxxy() = v; }) \
                    .def_property(                                                                                                                   \
                        "xxyx", [](Vector<T, 2> &self) { return self.xxyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xxyx() = v; }) \
                    .def_property(                                                                                                                   \
                        "xxyy", [](Vector<T, 2> &self) { return self.xxyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xxyy() = v; }) \
                    .def_property(                                                                                                                   \
                        "xyxx", [](Vector<T, 2> &self) { return self.xyxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xyxx() = v; }) \
                    .def_property(                                                                                                                   \
                        "xyxy", [](Vector<T, 2> &self) { return self.xyxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xyxy() = v; }) \
                    .def_property(                                                                                                                   \
                        "xyyx", [](Vector<T, 2> &self) { return self.xyyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xyyx() = v; }) \
                    .def_property(                                                                                                                   \
                        "xyyy", [](Vector<T, 2> &self) { return self.xyyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.xyyy() = v; }) \
                    .def_property(                                                                                                                   \
                        "yxxx", [](Vector<T, 2> &self) { return self.yxxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yxxx() = v; }) \
                    .def_property(                                                                                                                   \
                        "yxxy", [](Vector<T, 2> &self) { return self.yxxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yxxy() = v; }) \
                    .def_property(                                                                                                                   \
                        "yxyx", [](Vector<T, 2> &self) { return self.yxyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yxyx() = v; }) \
                    .def_property(                                                                                                                   \
                        "yxyy", [](Vector<T, 2> &self) { return self.yxyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yxyy() = v; }) \
                    .def_property(                                                                                                                   \
                        "yyxx", [](Vector<T, 2> &self) { return self.yyxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yyxx() = v; }) \
                    .def_property(                                                                                                                   \
                        "yyxy", [](Vector<T, 2> &self) { return self.yyxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yyxy() = v; }) \
                    .def_property(                                                                                                                   \
                        "yyyx", [](Vector<T, 2> &self) { return self.yyyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yyyx() = v; }) \
                    .def_property(                                                                                                                   \
                        "yyyy", [](Vector<T, 2> &self) { return self.yyyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) { self.yyyy() = v; })

void export_vector2(py::module &m) {
    OC_EXPORT_VECTOR2(float);
}