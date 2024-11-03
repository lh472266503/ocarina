//
// Created by ling.zhu on 2024/10/28.
//

#include "math/export_vector_func.h"
//#include "rhi/resources/resource.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector(py::module &m);
void export_matrix(py::module &m);
void export_scalar_cast(py::module &m) {
    using Tuple = std::tuple<uint, int, float>;
    traverse_tuple(Tuple{}, [&]<typename Src>(const Src &_, uint index) {
        traverse_tuple(Tuple{}, [&]<typename Dst>(const Dst &_, uint index) {
            if constexpr (std::is_same_v<Src, Dst>) {
                return;
            }
            string func_name = ocarina::format("as_{}", TypeDesc<Dst>::name());
            m.def(func_name.c_str(), [&](const Src &src) { return ocarina::bit_cast<Dst>(src); });
        });
    });
}
void export_math(py::module &m) {
    export_vector(m);
    export_scalar_cast(m);
    export_matrix(m);
}

void export_type(py::module &m);

void export_ast(py::module &m) {
    export_type(m);
}

void export_base_type(py::module &m);

void export_device(py::module &m);

void export_rhi(py::module &m) {
    export_device(m);
}

PYBIND11_MODULE(ocapi, m) {
    export_ast(m);
    export_math(m);
    export_rhi(m);
}