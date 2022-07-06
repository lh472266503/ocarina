//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "texture.h"
#include "core/concepts.h"

namespace ocarina {
class Context;
template<typename T>
class Buffer;

class Device : public concepts::Noncopyable {
public:
    class Impl : public concepts::Noncopyable {
    private:
        Context *_context{};
        friend class Device;

    public:
        explicit Impl(Context *ctx) : _context(ctx) {}
        [[nodiscard]] virtual handle_ty create_buffer(size_t size) noexcept = 0;
        virtual void destroy_buffer(handle_ty handle) noexcept = 0;
        virtual void destroy_texture(handle_ty handle) noexcept = 0;
        virtual void compile(const Function &function) noexcept = 0;
    };

    using Creator = Device::Impl *(Context *);
    using Deleter = void(Device::Impl *);
    using Handle = ocarina::unique_ptr<Device::Impl, Device::Deleter *>;

private:
    Handle _impl;

public:
    explicit Device(Handle impl) : _impl(std::move(impl)) {}
    [[nodiscard]] Context *context() const noexcept { return _impl->_context; }
    template<typename T>
    [[nodiscard]] Buffer<T> create_buffer(size_t size) noexcept {

    }
};
}// namespace ocarina