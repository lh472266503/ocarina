//
// Created by Zero on 2024/11/3.
//

#pragma once

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "core/stl.h"
#include "rhi/device.h"

namespace py = pybind11;
using namespace ocarina;

struct PythonExporter {
    py::module module;
};

