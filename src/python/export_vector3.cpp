//
// Created by ling.zhu on 2024/10/28.
//

#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_VECTOR3(type) \
    py::class_<ocarina::detail::VectorStorage<type, 3>>(m, "_VectorStorage"#type"3");

void export_vector3(py::module &m){
    OC_EXPORT_VECTOR3(float)
}