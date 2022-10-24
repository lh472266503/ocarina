//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "dsl/rtx_type.h"
#include "core/image_base.h"
#include "core/concepts.h"
#include "params.h"

namespace ocarina {
class Context;

template<typename T>
class Buffer;

template<typename T>
class Managed;

template<typename T>
class Shader;

class Stream;

class Image;

class Mesh;

class Accel;

class CommandVisitor;

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
        [[nodiscard]] virtual handle_ty create_image(uint2 res, PixelStorage pixel_storage) noexcept = 0;
        virtual void destroy_texture(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_shader(const Function &function) noexcept = 0;
        virtual void destroy_shader(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_accel() noexcept = 0;
        virtual void destroy_accel(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_stream() noexcept = 0;
        virtual void destroy_stream(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_mesh(const MeshParams &params) noexcept = 0;
        virtual void destroy_mesh(handle_ty handle) noexcept = 0;
        [[nodiscard]] Context *context() noexcept { return _context; }
        virtual void init_rtx() noexcept = 0;
        [[nodiscard]] virtual CommandVisitor *command_visitor() noexcept = 0;
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

    template<typename T = std::byte>
    [[nodiscard]] Buffer<T> create_buffer(size_t size) noexcept {
        return Buffer<T>(_impl.get(), size);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size) noexcept {
        return Managed<T>(_impl.get(), size);
    }

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] Mesh create_mesh(const VBuffer &v_buffer,
                                   const TBuffer &t_buffer,
                                   AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE,
                                   AccelGeomTag geom_tag = AccelGeomTag::DISABLE_ANYHIT) noexcept;// implement in mesh.h
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] Accel create_accel() noexcept;
    void init_rtx() noexcept { _impl->init_rtx(); }
    [[nodiscard]] Image create_image(uint2 res, PixelStorage storage) noexcept;
    template<typename T>
    [[nodiscard]] auto compile(const Kernel<T> &kernel, ShaderTag tag = CS) noexcept {
        return _create<Shader<T>>(kernel.function(), tag);
    }
};
}// namespace ocarina