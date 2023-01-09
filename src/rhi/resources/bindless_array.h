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
        [[nodiscard]] virtual size_t emplace_texture(handle_ty handle) noexcept = 0;
    };

public:
    explicit BindlessArray(Device::Impl *device);
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] size_t size() const noexcept { return impl()->size(); }
    [[nodiscard]] size_t alignment() const noexcept { return impl()->alignment(); }
    template<typename T>
    [[nodiscard]] size_t emplace(const BufferView<T> &buffer_view) noexcept {
        return impl()->emplace_buffer(buffer_view.head());
    }
    [[nodiscard]] size_t emplace(const RHITexture &texture) noexcept;
};

}// namespace ocarina