//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_VECTOR4(type) \
    py::class_<ocarina::detail::VectorStorage<type, 4>>(m, "_VectorStorage"#type"4");

void export_vector4(py::module &m){
    OC_EXPORT_VECTOR4(float)
}