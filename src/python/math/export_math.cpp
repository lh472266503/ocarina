//
// Created by Zero on 2024/11/3.
//

#include "export_vector_func.h"

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