//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

class RHITexture : public RHIResource {
public:
    class Impl {
    public:
        [[nodiscard]] virtual uint2 resolution() const noexcept = 0;
        [[nodiscard]] virtual PixelStorage pixel_storage() const noexcept = 0;
        [[nodiscard]] virtual handle_ty array_handle() const noexcept = 0;
        [[nodiscard]] virtual handle_ty tex_handle() const noexcept = 0;
        [[nodiscard]] virtual const void *handle_ptr() const noexcept = 0;

        /// for device side structure
        [[nodiscard]] virtual size_t data_size() const noexcept = 0;
        [[nodiscard]] virtual size_t data_alignment() const noexcept = 0;
        [[nodiscard]] virtual size_t max_member_size() const noexcept = 0;
    };

public:
    RHITexture() = default;
    explicit RHITexture(Device::Impl *device, uint2 res,
                        PixelStorage pixel_storage)
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage)) {}

    /// for dsl
    template<typename Output, typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    [[nodiscard]] auto sample(const U &u, const V &v) const noexcept {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<RHITexture>(),
                                                                             Variable::Tag::TEXTURE,
                                                                             memory_block());
        return make_expr<RHITexture>(uniform.expression()).sample<Output>(u, v);
    }

    template<typename Output, typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
        OC_NODISCARD auto sample(const UV &uv) const noexcept {
        return sample<Output>(uv.x, uv.y);
    }

    template<typename Target, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
    OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<RHITexture>(),
                                                                             Variable::Tag::TEXTURE,
                                                                             memory_block());
        return make_expr<RHITexture>(uniform.expression()).read<Target>(x, y);
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
        const ArgumentBinding &uniform = Function::current()->get_uniform_var(Type::of<RHITexture>(),
                                                                             Variable::Tag::TEXTURE,
                                                                             memory_block());
        make_expr<RHITexture>(uniform.expression()).write(x, y, elm);
    }

    template<typename XY, typename Val>
    requires(is_uint_vector2_v<expr_value_t<XY>>)
    void write(const XY &xy, const Val &elm) noexcept {
        write(xy.x, xy.y, elm);
    }

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] uint2 resolution() const noexcept { return impl()->resolution(); }
    [[nodiscard]] handle_ty array_handle() const noexcept { return impl()->array_handle(); }
    [[nodiscard]] handle_ty tex_handle() const noexcept { return impl()->tex_handle(); }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return impl()->handle_ptr(); }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return impl()->pixel_storage(); }
    [[nodiscard]] size_t data_size() const noexcept override { return impl()->data_size(); }
    [[nodiscard]] size_t data_alignment() const noexcept override { return impl()->data_alignment(); }
    [[nodiscard]] size_t max_member_size() const noexcept override { return impl()->max_member_size(); }
    [[nodiscard]] TextureUploadCommand *upload(const void *data) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), resolution(), pixel_storage(), false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), resolution(), pixel_storage(), false);
    }

    void upload_immediately(const void *data) const noexcept {
        upload_sync(data)->accept(*_device->command_visitor());
    }

    void download_immediately(void *data) const noexcept {
        download_sync(data)->accept(*_device->command_visitor());
    }
};

}// namespace ocarina