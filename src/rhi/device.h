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
#include "graphics_descriptions.h"

namespace ocarina {

class FileManager;

template<typename T, int... Dims>
class Buffer;

class ByteBuffer;

template<typename T, AccessMode mode = AOS, typename TBuffer = ByteBuffer>
class List;

template<typename T, AccessMode mode = AOS>
class ManagedList;

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
        FileManager *file_manager_{};
        friend class Device;

    public:
        explicit Impl(FileManager *ctx) : file_manager_(ctx) {}
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
        virtual void register_shared_buffer(void *&shared_handle, uint &gl_handle) noexcept = 0;
        virtual void register_shared_tex(void *&shared_handle, uint &gl_handle) noexcept = 0;
        virtual void mapping_shared_buffer(void *&shared_handle, handle_ty &handle) noexcept = 0;
        virtual void mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept = 0;
        virtual void unmapping_shared(void *&shared_handle) noexcept = 0;
        virtual void unregister_shared(void *&shared_handle) noexcept = 0;
        [[nodiscard]] FileManager *file_manager() noexcept { return file_manager_; }
        virtual void init_rtx() noexcept = 0;
        [[nodiscard]] virtual CommandVisitor *command_visitor() noexcept = 0;
    };

    using Creator = Device::Impl *(FileManager *);
    using Deleter = void(Device::Impl *);
    using Handle = ocarina::unique_ptr<Device::Impl, Device::Deleter *>;

private:
    Handle impl_;
    template<typename T, typename... Args>
    [[nodiscard]] auto _create(Args &&...args) const noexcept {
        return T(this->impl_.get(), std::forward<Args>(args)...);
    }

public:
    explicit Device(Handle impl) : impl_(std::move(impl)) {}
    [[nodiscard]] FileManager *file_manager() const noexcept { return impl_->file_manager_; }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, const string &name = "") const noexcept {
        return Buffer<T, Dims...>(impl_.get(), size, name);
    }

    [[nodiscard]] ByteBuffer create_byte_buffer(size_t size, const string &name = "") const noexcept;

    template<typename T, AccessMode mode = AOS>
    [[nodiscard]] List<T, mode> create_list(size_t size, const string &name = "") const noexcept; // implement in byte_buffer.h

    template<typename T, AccessMode mode = AOS>
    [[nodiscard]] ManagedList<T, mode> create_managed_list(size_t size, const string &name = "") const noexcept {
        return ManagedList<T, mode>(create_list<T, mode>(size, name));
    }

    template<typename T = std::byte, int... Dims>
    [[nodiscard]] Buffer<T, Dims...> create_buffer(size_t size, handle_ty stream) noexcept {
        return Buffer<T, Dims...>(impl_.get(), size, stream);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size) noexcept {
        return Managed<T>(impl_.get(), size);
    }

    template<typename T = std::byte>
    [[nodiscard]] Managed<T> create_managed(size_t size, handle_ty stream) noexcept {
        return Managed<T>(impl_.get(), size, stream);
    }

    template<typename VBuffer, typename TBuffer>
    [[nodiscard]] RHIMesh create_mesh(const VBuffer &v_buffer,
                                      const TBuffer &t_buffer,
                                      AccelUsageTag usage_tag = AccelUsageTag::FAST_TRACE,
                                      AccelGeomTag geom_tag = AccelGeomTag::DISABLE_ANYHIT) const noexcept;// implement in mesh.h
    [[nodiscard]] Stream create_stream() noexcept;
    [[nodiscard]] Accel create_accel() const noexcept;
    [[nodiscard]] BindlessArray create_bindless_array() const noexcept;
    void init_rtx() noexcept { impl_->init_rtx(); }
    [[nodiscard]] Texture create_texture(uint3 res, PixelStorage storage, const string &desc = "") const noexcept;
    [[nodiscard]] Texture create_texture(uint2 res, PixelStorage storage, const string &desc = "") const noexcept;
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