//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "core/image_base.h"
#include "core/concepts.h"

namespace ocarina {
class Context;

template<typename T>
class Buffer;

template<typename T>
class Shader;

enum ShaderTag : uint8_t {
    CS = 1 << 1,
    VS = 1 << 2,
    FS = 1 << 3,
    GS = 1 << 4,
    TS = 1 << 5
};

class Stream;

template<typename T>
class Texture;

class Device : public concepts::Noncopyable {
public:
    class Impl : public concepts::Noncopyable {
    protected:
        Context *_context{};
        friend class Device;

    public:
        explicit Impl(Context *ctx) : _context(ctx) {}
        [[nodiscard]] virtual handle_ty create_buffer(size_t size) noexcept = 0;
        virtual void destroy_buffer(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_texture(uint2 res, PixelStorage pixel_storage) noexcept = 0;
        virtual void destroy_texture(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_shader(const Function &function) noexcept = 0;
        virtual void destroy_shader(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_accel() noexcept = 0;
        virtual void destroy_accel(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_stream() noexcept = 0;
        virtual void destroy_stream(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_mesh(handle_ty v_handle,handle_ty t_handle,
                                                    uint v_stride,uint t_count) noexcept = 0;
        virtual void destroy_mesh(handle_ty handle) noexcept = 0;
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

    template<typename T>
    [[nodiscard]] Texture<T> create_texture(uint2 res) noexcept {
        return _create<Texture<T>>(res, PixelStorageImpl<T>::storage);
    }
    template<typename T>
    [[nodiscard]] auto compile(const Kernel<T> &kernel, ShaderTag tag = CS) noexcept {
        return _create<Shader<T>>(kernel.function(), tag);
    }
};
}// namespace ocarina