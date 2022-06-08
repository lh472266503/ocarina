//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"

namespace ocarina {
class Context;
class Device {
protected:
    Context *_context{};

public:
    explicit Device(Context *ctx) : _context(ctx) {}
    [[nodiscard]] Context *context() const noexcept { return _context; }
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;
    virtual void destroy_buffer(uint64_t handle) noexcept = 0;
    [[nodiscard]] virtual uint64_t create_stream() noexcept = 0;
    virtual void destroy_stream(uint64_t handle) noexcept = 0;
    virtual void compile(Function function) noexcept = 0;
};
}// namespace ocarina