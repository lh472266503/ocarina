//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "buffer.h"
#include "texture.h"
#include "core/concepts.h"

namespace ocarina {
class Context;
class Device : public concepts::Noncopyable {
protected:
    Context *_context{};

public:
    using Creator = Device *(Context *);
    using Deleter = void(Device *);
    using Handle = ocarina::unique_ptr<Device, Device::Deleter*>;

public:
    explicit Device(Context *ctx) : _context(ctx) {}
    [[nodiscard]] Context *context() const noexcept { return _context; }
    [[nodiscard]] virtual uint64_t create_buffer(size_t size_bytes) noexcept = 0;

    virtual void destroy_buffer(uint64_t handle) noexcept = 0;
    virtual void compile(const Function &function) noexcept = 0;
};
}// namespace ocarina