//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

class Texture : public RHIResource {
public:
    class Impl {
    public:
        [[nodiscard]] virtual uint3 resolution() const noexcept = 0;
        [[nodiscard]] virtual PixelStorage pixel_storage() const noexcept = 0;
        [[nodiscard]] virtual handle_ty array_handle() const noexcept = 0;
        [[nodiscard]] virtual handle_ty tex_handle() const noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    Texture() = default;
    explicit Texture(Device::Impl *device, uint3 res,
                        PixelStorage pixel_storage)
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage)) {}

    /// for dsl
    [[nodiscard]] const Expression *expression() const noexcept override {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<decltype(*this)>(),
                                                                              Variable::Tag::TEXTURE,
                                                                              memory_block());
        return uniform.expression();
    }

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
        [[nodiscard]] auto sample(uint channel_num, const U &u, const V &v) const noexcept {
        return make_expr<Texture>(expression()).sample(channel_num, u, v);
    }

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
        OC_NODISCARD auto sample(uint channel_num, const UV &uv) const noexcept {
        return sample(channel_num, uv.x, uv.y);
    }

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V>)
        [[nodiscard]] auto sample(uint channel_num, const U &u, const V &v, const W &w) const noexcept {
        return make_expr<Texture>(expression()).sample(channel_num, u, v, w);
    }

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
        OC_NODISCARD auto sample(uint channel_num, const UVW &uvw) const noexcept {
        return sample(channel_num, uvw.x, uvw.y, uvw.z);
    }

    template<typename Target, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
        OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        return make_expr<Texture>(expression()).read<Target>(x, y);
    }

    template<typename Target, typename XY>
    requires(is_int_vector2_v<expr_value_t<XY>> ||
             is_uint_vector2_v<expr_value_t<XY>> &&
                 (is_uchar_element_expr_v<Target> || is_float_element_expr_v<Target>))
    OC_NODISCARD auto read(const XY &xy) const noexcept {
        return read<Target>(xy.x, xy.y);
    }

    template<typename X, typename Y, typename Val>
    requires(is_all_integral_expr_v<X, Y> &&
             (is_uchar_element_expr_v<Val> || is_float_element_expr_v<Val>))
    void write(const X &x, const Y &y, const Val &elm) noexcept {
        make_expr<Texture>(expression()).write(x, y, elm);
    }

    template<typename XY, typename Val>
    requires(is_uint_vector2_v<expr_value_t<XY>>) void write(const XY &xy, const Val &elm) noexcept {
        write(xy.x, xy.y, elm);
    }

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] Impl *operator->() noexcept { return impl(); }
    [[nodiscard]] const Impl *operator->() const noexcept { return impl(); }
    [[nodiscard]] handle_ty array_handle() const noexcept { return impl()->array_handle(); }
    [[nodiscard]] handle_ty tex_handle() const noexcept { return impl()->tex_handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] size_t data_size() const noexcept override { return impl()->data_size(); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return impl()->data_alignment(); }
    [[nodiscard]] size_t max_member_size() const noexcept override { return impl()->max_member_size(); }
    [[nodiscard]] TextureUploadCommand *upload(const void *data) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), impl()->resolution(),
                                            impl()->pixel_storage(), true);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), impl()->resolution(),
                                            impl()->pixel_storage(), false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), impl()->resolution(),
                                              impl()->pixel_storage(), true);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), impl()->resolution(),
                                              impl()->pixel_storage(), false);
    }

    void upload_immediately(const void *data) const noexcept {
        upload_sync(data)->accept(*_device->command_visitor());
    }

    void download_immediately(void *data) const noexcept {
        download_sync(data)->accept(*_device->command_visitor());
    }
};

}// namespace ocarina