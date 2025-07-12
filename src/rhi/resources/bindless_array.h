//
// Created by Zero on 03/11/2022.
//

#pragma once

#include "resource.h"
#include "dsl/type_trait.h"
#include "dsl/var.h"
#include "rhi/command.h"
#include "rhi/command_queue.h"
#include "rhi/device.h"

namespace ocarina {

class BindlessArray : public RHIResource {
public:
    class Impl {
    public:
        static constexpr auto c_max_slot_num = 50000;
        [[nodiscard]] virtual size_t emplace_buffer(handle_ty handle, uint offset_in_byte,
                                                    size_t size_in_byte) noexcept = 0;
        virtual void remove_buffer(handle_ty index) noexcept = 0;
        [[nodiscard]] virtual size_t emplace_texture(handle_ty handle) noexcept = 0;
        virtual void remove_texture(handle_ty index) noexcept = 0;
        virtual void set_buffer(handle_ty index, handle_ty handle, uint offset_in_byte,
                                size_t size_in_byte) noexcept = 0;
        [[nodiscard]] virtual ByteBufferDesc buffer_view(uint index) const noexcept = 0;
        virtual void set_texture(handle_ty index, handle_ty handle) noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_buffer_handles(bool async) const noexcept = 0;
        [[nodiscard]] virtual BufferUploadCommand *upload_texture_handles(bool async) const noexcept = 0;
        virtual void prepare_slotSOA(Device &device) noexcept = 0;
        virtual CommandList update_slotSOA(bool async) noexcept = 0;
        [[nodiscard]] virtual size_t buffer_num() const noexcept = 0;
        [[nodiscard]] virtual size_t texture_num() const noexcept = 0;
        [[nodiscard]] virtual size_t buffer_slots_size() const noexcept = 0;
        [[nodiscard]] virtual size_t tex_slots_size() const noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    BindlessArray() = default;
    explicit BindlessArray(Device::Impl *device);
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(handle_); }
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(handle_); }
    [[nodiscard]] const Impl *operator->() const noexcept { return impl(); }
    [[nodiscard]] Impl *operator->() noexcept { return impl(); }
    void prepare_slotSOA(Device &device) noexcept { impl()->prepare_slotSOA(device); }

    [[nodiscard]] CommandList update_slotSOA() noexcept {
        return impl()->update_slotSOA(true);
    }

    [[nodiscard]] CommandList update_slotSOA_sync() noexcept {
        return impl()->update_slotSOA(false);
    }

    [[nodiscard]] static constexpr size_t max_slot_num() noexcept { return Impl::c_max_slot_num; }

    /// for device side structure
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] size_t max_member_size() const noexcept override { return impl()->max_member_size(); }
    [[nodiscard]] size_t data_size() const noexcept override { return impl()->data_size(); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return impl()->data_alignment(); }

    template<typename T>
    requires is_buffer_or_view_v<T>
    [[nodiscard]] size_t emplace(const T &buffer) noexcept {
        return impl()->emplace_buffer(buffer.head(), buffer.offset_in_byte(),
                                      buffer.size_in_byte());
    }
    template<typename T>
    requires is_buffer_or_view_v<T>
    void set_buffer(handle_ty index, const T &buffer) noexcept {
        impl()->set_buffer(index, buffer.head(), buffer.offset_in_byte(),
                           buffer.size_in_byte());
    }
    size_t emplace(const Texture &texture) noexcept;
    void set_texture(handle_ty index, const Texture &texture) noexcept;
    [[nodiscard]] uint buffer_num() const noexcept;
    [[nodiscard]] uint texture_num() const noexcept;
    [[nodiscard]] CommandList upload_handles(bool async = true) noexcept;

    /// for dsl
    [[nodiscard]] const Expression *expression() const noexcept override {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                               Variable::Tag::BINDLESS_ARRAY,
                                                                                               memory_block());
        return captured_resource.expression();
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] BindlessArrayTexture tex_var(Index &&index) const noexcept {
        return expr().tex_var(OC_FORWARD(index),
                              typeid(*this).name(),
                              texture_num());
    }

    template<typename T, typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] BindlessArrayBuffer<T> buffer_var(Index &&index) const noexcept {
        return expr().buffer_var<T>(OC_FORWARD(index),
                                    typeid(*this).name(),
                                    buffer_num());
    }

    template<typename T>
    [[nodiscard]] BufferView<T> buffer_view(uint index) const noexcept {
        ByteBufferDesc buffer_desc = impl()->buffer_view(index);
        auto size = buffer_desc.size_in_byte() / sizeof(T);
        auto offset = 0 / sizeof(T);
        return BufferView<T>(buffer_desc.head(), offset, size, offset + size);
    }

    [[nodiscard]] ByteBufferView byte_buffer_view(uint index) const noexcept;

    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] BindlessArrayByteBuffer byte_buffer_var(Index &&index) const noexcept {
        return expr().byte_buffer_var(OC_FORWARD(index),
                                      typeid(*this).name(),
                                      buffer_num());
    }

    template<typename Elm, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] SOAViewVar<Elm, BindlessArrayByteBuffer> soa_view_var(Index &&index) noexcept {
        return byte_buffer_var(OC_FORWARD(index)).template soa_view_var<Elm>();
    }

    template<typename Elm, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] AOSViewVar<Elm, BindlessArrayByteBuffer> aos_view_var(Index &&index) noexcept {
        return byte_buffer_var(OC_FORWARD(index)).template aos_view_var<Elm>();
    }

    [[nodiscard]] Var<BindlessArray> var() const noexcept {
        return Var<BindlessArray>(expression());
    }

    [[nodiscard]] Expr<BindlessArray> expr() const noexcept {
        return make_expr<BindlessArray>(expression());
    }
};

}// namespace ocarina