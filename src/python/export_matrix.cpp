//
// Created by ling.zhu on 2024/10/30.
//

#include "export_vector_func.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

template<size_t N, size_t M, typename Module>
void export_matrix_func(py::module &m, Module &mt) {
#define OC_EXPORT_MATRIX_FUNC(func) \
    m.def(#func, [&](Matrix<N, M> mat) { return ocarina::func(mat); });

    if constexpr (N == M) {
        OC_EXPORT_MATRIX_FUNC(inverse)
        OC_EXPORT_MATRIX_FUNC(transpose)
        OC_EXPORT_MATRIX_FUNC(determinant)
    }

    OC_EXPORT_MATRIX_FUNC(rcp)
    OC_EXPORT_MATRIX_FUNC(abs)
    OC_EXPORT_MATRIX_FUNC(sqrt)
    OC_EXPORT_MATRIX_FUNC(sqr)
    OC_EXPORT_MATRIX_FUNC(sign)
    OC_EXPORT_MATRIX_FUNC(cos)
    OC_EXPORT_MATRIX_FUNC(sin)
    OC_EXPORT_MATRIX_FUNC(tan)
    OC_EXPORT_MATRIX_FUNC(cosh)
    OC_EXPORT_MATRIX_FUNC(sinh)
    OC_EXPORT_MATRIX_FUNC(tanh)
    OC_EXPORT_MATRIX_FUNC(log)
    OC_EXPORT_MATRIX_FUNC(log2)
    OC_EXPORT_MATRIX_FUNC(log10)
    OC_EXPORT_MATRIX_FUNC(exp)
    OC_EXPORT_MATRIX_FUNC(exp2)
    OC_EXPORT_MATRIX_FUNC(asin)
    OC_EXPORT_MATRIX_FUNC(acos)
    OC_EXPORT_MATRIX_FUNC(atan)
    OC_EXPORT_MATRIX_FUNC(asinh)
    OC_EXPORT_MATRIX_FUNC(acosh)
    OC_EXPORT_MATRIX_FUNC(atanh)
    OC_EXPORT_MATRIX_FUNC(floor)
    OC_EXPORT_MATRIX_FUNC(ceil)
    OC_EXPORT_MATRIX_FUNC(degrees)
    OC_EXPORT_MATRIX_FUNC(radians)
    OC_EXPORT_MATRIX_FUNC(round)
    OC_EXPORT_MATRIX_FUNC(isnan)
    OC_EXPORT_MATRIX_FUNC(isinf)
    OC_EXPORT_MATRIX_FUNC(fract)

#undef OC_EXPORT_MATRIX_FUNC

}

template<size_t N, size_t M>
auto export_matrix_base(py::module &m) {
    string cls_name = ocarina::format("float{}x{}", N, M);
    auto mt = py::class_<Matrix<N, M>>(m, cls_name.c_str());
    mt.def("__getitem__", [](Matrix<N, M> &self, size_t i) { return &self[i]; }, py::return_value_policy::reference_internal);
    mt.def("__setitem__", [](Matrix<N, M> &self, size_t i, Vector<float, M> k) { self[i] = k; });
    mt.def("__neg__", [](Matrix<N, M> &self) { return -self; });
    mt.def("__mul__", [](Matrix<N, M> &self, float s) { return self * s; });
    mt.def("__rmul__", [](Matrix<N, M> &self, float s) { return self * s; });
    mt.def("__mul__", [](Matrix<N, M> &self, ocarina::Vector<float, N> v) { return self * v; });
    mt.def("__mul__", [](Matrix<N, M> &self, ocarina::Matrix<M, N> rhs) { return self * rhs; });
    mt.def("__truediv__", [](Matrix<N, M> &self,  float s) { return self / s; });
    mt.def("__add__", [](Matrix<N, M> &self, ocarina::Matrix<N, M> rhs) { return self + rhs; });
    mt.def("__sub__", [](Matrix<N, M> &self, ocarina::Matrix<N, M> rhs) { return self - rhs; });

    auto make_func_name = "make_" + cls_name;
    auto export_constructor = [&]<typename Func>(const Func &func) {
        mt.def(py::init(func));
        m.def(make_func_name.c_str(), func);
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
    
    return mt;
}

template<size_t N, size_t M>
void export_matrix_(py::module &m) {
    auto mt = export_matrix_base<N, M>(m);
    export_matrix_func<N, M>(m, mt);
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