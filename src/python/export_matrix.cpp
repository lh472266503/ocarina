//
// Created by ling.zhu on 2024/10/30.
//

#include "export_vector_func.h"

namespace py = pybind11;
using namespace ocarina;

template<size_t N, size_t M>
void export_matrix_(py::module &m) {
    string cls_name = ocarina::format("float{}x{}", N, M);
    auto mt = py::class_<Matrix<N, M>>(m, cls_name.c_str());
    mt.def("__getitem__", [](Matrix<N, M> &self, size_t i) { return &self[i]; }, py::return_value_policy::reference_internal);
    mt.def("__setitem__", [](Matrix<N, M> &self, size_t i, Vector<float, M> k) { self[i] = k; });

    auto export_constructor = [&]<typename Func>(const Func &func) {
        auto func_name = "make_" + cls_name;
        mt.def(py::init(func));
        mt.def(func_name.c_str(), func);
    };

    export_constructor([](float a) { return Matrix<N, M>(a); });

    export_constructor([](const std::array<float, M * N> &a) {
        return [&]<size_t... i>(std::index_sequence<i...>) {
            return Matrix<N, M>(a[i]...);
        }(std::make_index_sequence<M * N>());
    });

    export_constructor([](std::array<Vector<float, M>, N> a) {
        return [&]<size_t... i>(std::index_sequence<i...>) {
            return Matrix<N, M>(a[i]...);
        }(std::make_index_sequence<N>());
    });

    export_constructor([](std::array<array<float, M>, N> a) {
        return [&]<size_t... i>(std::index_sequence<i...>) {
            return Matrix<N, M>(Vector<float, M>(a[i].data())...);
        }(std::make_index_sequence<N>());
    });

    mt.def("__repr__", [](Matrix<N, M> &self) {
        return to_str(self);
    });

    
}

void export_matrix(py::module &m) {
    export_matrix_<2, 2>(m);
    export_matrix_<2, 3>(m);
    export_matrix_<2, 4>(m);

    export_matrix_<3, 2>(m);
    export_matrix_<3, 3>(m);
    export_matrix_<3, 4>(m);

    export_matrix_<4, 2>(m);
    export_matrix_<4, 3>(m);
    export_matrix_<4, 4>(m);
}