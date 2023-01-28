//
// Created by Zero on 03/11/2022.
//

#pragma once

#include "resource.h"
#include "dsl/type_trait.h"
#include "dsl/var.h"
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
        [[nodiscard]] virtual BufferUploadCommand *upload_buffer_handles_sync() const noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_texture_handles_sync() const noexcept = 0;
        virtual void prepare_slotSOA(Device &device) noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    BindlessArray() = default;
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
    size_t emplace(const RHITexture &texture) noexcept;
    void remove_buffer(handle_ty index) noexcept;
    void remove_texture(handle_ty index) noexcept;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles() noexcept;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles() noexcept;
    [[nodiscard]] BufferUploadCommand *upload_buffer_handles_sync() noexcept;
    [[nodiscard]] BufferUploadCommand *upload_texture_handles_sync() noexcept;

    /// for dsl
    [[nodiscard]] const Expression *expression() const noexcept override {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::BINDLESS_ARRAY,
                                                                              memory_block());
        return uniform.expression();
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] BindlessArrayTexture tex(Index &&index) const noexcept {
        return make_expr<BindlessArray>(expression()).tex(OC_FORWARD(index));
    }

    template<typename T, typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] BindlessArrayBuffer<T> buffer(Index &&index) const noexcept {
        return make_expr<BindlessArray>(expression()).buffer<T>(OC_FORWARD(index));
    }

    [[nodiscard]] Var<BindlessArray> var() const noexcept {
        return Var<BindlessArray>(expression());
    }
};

}// namespace ocarina