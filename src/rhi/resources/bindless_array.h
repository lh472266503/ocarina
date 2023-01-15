//
// Created by Zero on 03/11/2022.
//

#pragma once

#include "resource.h"
#include "dsl/type_trait.h"
#include "rhi/command.h"
#include "rhi/device.h"

namespace ocarina {

class BindlessArray : public RHIResource {
public:
    class Impl {
    public:
        [[nodiscard]] virtual size_t emplace_buffer(handle_ty handle) noexcept = 0;
        virtual void remove_buffer(handle_ty index) noexcept = 0;
        [[nodiscard]] virtual size_t emplace_texture(handle_ty handle) noexcept = 0;
        virtual void remove_texture(handle_ty index) noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_buffer_handles() const noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_texture_handles() const noexcept = 0;
        virtual void prepare_slotSOA(Device &device) noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    explicit BindlessArray(Device::Impl *device);
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    void prepare_slotSOA(Device &device) noexcept { impl()->prepare_slotSOA(device); }

    /// for device side structure
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] size_t max_member_size() const noexcept override { return impl()->max_member_size(); }
    [[nodiscard]] size_t data_size() const noexcept override { return impl()->data_size(); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return impl()->data_alignment(); }

    template<typename T>
    requires is_buffer_or_view_v<T>
    [[nodiscard]] size_t emplace(const T &buffer) noexcept {
        return impl()->emplace_buffer(buffer.head());
    }
    [[nodiscard]] size_t emplace(const RHITexture &texture) noexcept;
    void remove_buffer(handle_ty index) noexcept;
    void remove_texture(handle_ty index) noexcept;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles() noexcept;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles() noexcept;
};

}// namespace ocarina