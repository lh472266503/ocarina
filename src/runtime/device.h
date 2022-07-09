//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "core/concepts.h"

namespace ocarina {
class Context;

template<typename T>
class Buffer;

template<size_t dimension, typename... Args>
class Shader;

class Stream;

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
        [[nodiscard]] virtual handle_ty create_stream() noexcept = 0;
        virtual void destroy_stream(handle_ty handle) noexcept = 0;
        virtual void compile(const Function &function) noexcept = 0;
    };

    using Creator = Device::Impl *(Context *);
    using Deleter = void(Device::Impl *);
    using Handle = ocarina::unique_ptr<Device::Impl, Device::Deleter *>;

private:
    Handle _impl;
    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) noexcept {
        return T(this->_impl.get(), std::forward<Args>(args)...);
    }

public:
    explicit Device(Handle impl) : _impl(std::move(impl)) {}
    [[nodiscard]] Context *context() const noexcept { return _impl->_context; }
    template<typename T>
    [[nodiscard]] Buffer<T> create_buffer(size_t size) noexcept {
        return Buffer<T>(_impl.get(), size);
    }
    [[nodiscard]] Stream create_stream() noexcept;
};
}// namespace ocarina