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
#include "core/thread_pool.h"
#include "params.h"

namespace ocarina {
class FileManager;

template<typename T, int... Dims>
class Buffer;

template<typename T>
class Managed;

class BindlessArray;

template<typename T>
class Shader;

class Stream;

class Texture;

class RHIMesh;

class Accel;

class CommandVisitor;

class Device : public concepts::Noncopyable {
public:
    class Impl : public concepts::Noncopyable {
    protected:
        FileManager *_context{};
        friend class Device;

    public:
        explicit Impl(FileManager *ctx) : _context(ctx) {}
        [[nodiscard]] virtual handle_ty create_buffer(size_t size, const string &desc) noexcept = 0;
        virtual void destroy_buffer(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_texture(uint3 res, PixelStorage pixel_storage,
                                                       uint level_num, const string &desc) noexcept = 0;
        virtual void destroy_texture(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_shader(const Function &function) noexcept = 0;
        virtual void destroy_shader(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_accel() noexcept = 0;
        virtual void destroy_accel(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_stream() noexcept = 0;
        virtual void destroy_stream(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_mesh(const MeshParams &params) noexcept = 0;
        virtual void destroy_mesh(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual handle_ty create_bindless_array() noexcept = 0;
        virtual void destroy_bindless_array(handle_ty handle) noexcept = 0;
        [[nodiscard]] FileManager *file_manager() noexcept { return _context; }
        virtual void init_rtx() noexcept = 0;
        [[nodiscard]] virtual CommandVisitor *command_visitor() noexcept = 0;
    };

    using Creator = Device::Impl *(FileManager *);
    using Deleter = void(Device::Impl *);
    using Handle = ocarina::unique_ptr<Device::Impl, Device::Deleter *>;

private:
    Handle _impl;
    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) const noexcept {
        return T(this->_impl.get(), std::forward<Args>(args)...);
    }

public:
    explicit Device(Handle impl) : _impl(std::move(impl)) {}
    [[nodiscard]] FileManager *file_manager() const noexcept { return _impl->_context; }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, const string &desc = "") noexcept {
        return Buffer<T, Dims...>(_impl.get(), size, desc);
    }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, handle_ty stream) noexcept {
        return Buffer<T, Dims...>(_impl.get(), size, stream);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size) noexcept {
        return Managed<T>(_impl.get(), size);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size, handle_ty stream) noexcept {
        return Managed<T>(_impl.get(), size, stream);
    }

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] RHIMesh create_mesh(const VBuffer &v_buffer,
                                      const TBuffer &t_buffer,
                                      AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE,
                                      AccelGeomTag geom_tag = AccelGeomTag::DISABLE_ANYHIT) noexcept;// implement in mesh.h
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] Accel create_accel() noexcept;
    [[nodiscard]] BindlessArray create_bindless_array() noexcept;
    void init_rtx() noexcept { _impl->init_rtx(); }
    [[nodiscard]] Texture create_texture(uint3 res, PixelStorage storage, const string &desc = "") noexcept;
    [[nodiscard]] Texture create_texture(uint2 res, PixelStorage storage, const string &desc = "") noexcept;
    template<typename T>
    [[nodiscard]] auto compile(const Kernel<T> &kernel, const string &shader_desc = "", ShaderTag tag = CS) const noexcept {
        OC_INFO_FORMAT("compile shader : {}", shader_desc.c_str());
        kernel.function()->set_description(shader_desc);
        return _create<Shader<T>>(kernel.function(), tag);
    }
    template<typename T>
    [[nodiscard]] auto async_compile(Kernel<T> &&kernel, const string &shader_desc = "", ShaderTag tag = CS) const noexcept {
        return async([=, this, kernel = ocarina::move(kernel)] {
            return compile(kernel, shader_desc, tag);
        });
    }
};
}// namespace ocarina