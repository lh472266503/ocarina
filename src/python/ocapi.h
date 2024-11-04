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
    UP<py::class_<Device, concepts::Noncopyable>> m_device;
};

struct Context {
    [[nodiscard]] static Context &instance() noexcept;
    Context() {
        OC_INFO("ocapi load!");
    }
    ~Context() {
        OC_INFO("ocapi unload!");
    }
};