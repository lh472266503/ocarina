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
        [[nodiscard]] virtual size_t size() const noexcept = 0;
        [[nodiscard]] virtual size_t alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t emplace_buffer(handle_ty handle) noexcept = 0;
        virtual void remove_buffer(handle_ty index) noexcept = 0;
        virtual void remove_texture(handle_ty index) noexcept = 0;
        [[nodiscard]] virtual size_t emplace_texture(handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_buffer_handles() const noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_texture_handles() const noexcept = 0;
    };

public:
    explicit BindlessArray(Device::Impl *device);
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] size_t size() const noexcept { return impl()->size(); }
    [[nodiscard]] size_t alignment() const noexcept { return impl()->alignment(); }
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