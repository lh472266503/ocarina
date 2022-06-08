//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"

namespace ocarina {

class Device {
protected:
public:
    Device();
    // buffer
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    virtual void destroy_buffer(uint64_t handle) noexcept = 0;
    // stream
    [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
    virtual void destroy_stream(uint64_t handle) noexcept = 0;

    virtual void compile(Function function) noexcept = 0;
};
}// namespace ocarina