//
// Created by Zero on 2024/11/2.
//

#include "export_vector_func.h"
#include "swizzle_inl/export_swizzle2.inl.h"
#include "swizzle_inl/export_swizzle3.inl.h"
#include "swizzle_inl/export_swizzle4.inl.h"
#include "rhi/resources/buffer.h"

namespace py = pybind11;
using namespace ocarina;

template<typename T, size_t N>
auto export_vector_type(PythonExporter &exporter) {
    auto &m = exporter.module;
    using vector_type = Vector<T, N>;
    static string type_str = string(TypeDesc<vector_type>::name());
    static string base_type_str = "_VectorStorage" + type_str;
    static string make_str = "make_" + type_str;
    auto _ = py::class_<ocarina::detail::VectorStorage<T, N>>(m, base_type_str.c_str());
    auto ret = export_pod_type<vector_type, ocarina::detail::VectorStorage<T, N>>(exporter)
                   .def(py::init<>())
                   .def(py::init<T>())
                   .def("__getitem__", [](const vector_type &self, size_t i) { return self[i]; })
                   .def("__setitem__", [](vector_type &self, size_t i, T k) { self[i] = k; })
                   .def_readwrite("x", &vector_type::x)
                   .def_readwrite("y", &vector_type::y);
    if constexpr (N > 2) { ret.def_readwrite("z", &vector_type::z); }
    if constexpr (N > 3) { ret.def_readwrite("w", &vector_type::w); }
    if constexpr (N == 2) {
        ret.def(py::init<T, T>());
        m.def(make_str.c_str(), [](T a, T b) {
            return vector_type{a, b};
        });
        m.def(make_str.c_str(), [](vector_type v) {
            return v;
        });
    } else if constexpr (N == 3) {
        ret.def(py::init<T, T, T>());
        m.def(make_str.c_str(), [](T a, T b, T c) {
            return vector_type{a, b, c};
        });
        m.def(make_str.c_str(), [](vector_type v) {
            return v;
        });
    } else if constexpr (N == 4) {
        ret.def(py::init<T, T, T, T>());
        m.def(make_str.c_str(), [](T a, T b, T c, T d) {
            return vector_type{a, b, c, d};
        });
        m.def(make_str.c_str(), [](vector_type v) {
            return v;
        });
    }
    return ret;
}

void export_vector(PythonExporter &exporter) {
    auto &m = exporter.module;
#define OC_EXPORT_VECTOR(T, N) auto m##T##N = export_vector_type<T, N>(exporter);
    OC_EXPORT_VECTOR(bool, 2)
    OC_EXPORT_VECTOR(bool, 3)
    OC_EXPORT_VECTOR(bool, 4)

    OC_EXPORT_VECTOR(uint, 2)
    OC_EXPORT_VECTOR(uint, 3)
    OC_EXPORT_VECTOR(uint, 4)

    OC_EXPORT_VECTOR(int, 2)
    OC_EXPORT_VECTOR(int, 3)
    OC_EXPORT_VECTOR(int, 4)

    OC_EXPORT_VECTOR(float, 2)
    OC_EXPORT_VECTOR(float, 3)
    OC_EXPORT_VECTOR(float, 4)

#undef OC_EXPORT_VECTOR

#define OC_EXPORT_VECTOR_FUNC(T, N) \
    export_swizzle##N<T>(m##T##N);  \
    export_vector_func<T, N>(m##T##N, exporter);

    OC_EXPORT_VECTOR_FUNC(bool, 2)
    OC_EXPORT_VECTOR_FUNC(bool, 3)
    OC_EXPORT_VECTOR_FUNC(bool, 4)

    OC_EXPORT_VECTOR_FUNC(uint, 2)
    OC_EXPORT_VECTOR_FUNC(uint, 3)
    OC_EXPORT_VECTOR_FUNC(uint, 4)

    OC_EXPORT_VECTOR_FUNC(int, 2)
    OC_EXPORT_VECTOR_FUNC(int, 3)
    OC_EXPORT_VECTOR_FUNC(int, 4)

    OC_EXPORT_VECTOR_FUNC(float, 2)
    OC_EXPORT_VECTOR_FUNC(float, 3)
    OC_EXPORT_VECTOR_FUNC(float, 4)

#undef OC_EXPORT_VECTOR_FUNC
}