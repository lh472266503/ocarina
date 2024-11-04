//
// Created by Zero on 2024/11/3.
//

#include "python/ocapi.h"
#include "python/common.h"
#include "rhi/resources/buffer.h"
#include "math/base.h"
#include "ast/type_desc.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector(PythonExporter &exporter);
void export_matrix(PythonExporter &exporter);
void export_scalar_op(PythonExporter &exporter) {
    auto &m = exporter.module;
    using Tuple = std::tuple<uint, int, float>;
    traverse_tuple(Tuple{}, [&]<typename Src>(const Src &_, uint index) {
        traverse_tuple(Tuple{}, [&]<typename Dst>(const Dst &_, uint index) {
            if constexpr (std::is_same_v<Src, Dst>) {
                return;
            }
            string func_name = ocarina::format("as_{}", TypeDesc<Dst>::name());
            m.def(func_name.c_str(), [&](const Src &src) { return ocarina::bit_cast<Dst>(src); });
        });
        export_buffer<Src>(exporter);
    });
}

void export_math(PythonExporter &exporter) {
    Context::instance();
    export_vector(exporter);
    export_scalar_op(exporter);
    export_matrix(exporter);
}