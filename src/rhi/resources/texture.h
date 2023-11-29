//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

namespace detail {
[[nodiscard]] constexpr uint compute_mip_level_num(uint3 res, uint request_level_num) noexcept {
    uint max_size = std::max({res.x, res.y, res.z});
    auto max_levels = 0u;
    while (max_size != 0u) {
        max_size >>= 1u;
        max_levels++;
    }
    return request_level_num == 0 ? max_levels : std::min(request_level_num, max_levels);
}
}// namespace detail

class Texture : public RHIResource {
public:
    class Impl {
    public:
        virtual ~Impl() = default;
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
                     PixelStorage pixel_storage, uint level_num = 1u,
                     const string &desc = "")
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage,
                                             detail::compute_mip_level_num(res, level_num), desc)) {}

    /// for dsl
    [[nodiscard]] const Expression *expression() const noexcept override {
        const CapturedVar &captured_var = Function::current()->get_captured_var(Type::of<decltype(*this)>(),
                                                                               Variable::Tag::TEXTURE,
                                                                               memory_block());
        return captured_var.expression();
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
    requires(is_uint_vector2_v<expr_value_t<XY>>)
    void write(const XY &xy, const Val &elm) noexcept {
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
    [[nodiscard]] TextureUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return TextureUploadCommand::create(data, array_handle(), impl()->resolution(),
                                            impl()->pixel_storage(), async);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return upload(data, false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data, bool async = true) const noexcept {
        return TextureDownloadCommand::create(data, array_handle(), impl()->resolution(),
                                              impl()->pixel_storage(), async);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return download(data, false);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg>
    [[nodiscard]] BufferToTextureCommand *copy_from(const Arg &buffer, size_t buffer_offset, bool async = true) const noexcept {
        return BufferToTextureCommand::create(buffer.handle(), buffer_offset * buffer.element_size(), array_handle(),
                                              impl()->pixel_storage(),
                                              impl()->resolution(), 0, async);
    }

    void upload_immediately(const void *data) const noexcept {
        upload_sync(data)->accept(*_device->command_visitor());
    }

    void download_immediately(void *data) const noexcept {
        download_sync(data)->accept(*_device->command_visitor());
    }
};

}// namespace ocarina