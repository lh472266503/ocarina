//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "dynamic_module.h"
#include "device.h"
#include "core/platform.h"

namespace ocarina {
struct Context::Impl {
    fs::path runtime_directory;
    fs::path cache_directory;
    Device::Handle device;
    bool use_cache{true};
    ocarina::map<string, DynamicModule> modules;
    Impl() : device(Device::Handle(nullptr, nullptr)) {}
};
}// namespace ocarina