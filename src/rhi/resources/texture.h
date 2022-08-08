//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "rhi/command.h"

namespace ocarina {

template<typename T>
class Texture : public RHIResource {
public:
    class Impl {
    public:
        [[nodiscard]] virtual uint2 resolution() const noexcept = 0;
        [[nodiscard]] virtual PixelStorage pixel_storage() const noexcept = 0;
        [[nodiscard]] virtual handle_ty write_handle() const noexcept = 0;
        [[nodiscard]] virtual handle_ty read_handle() const noexcept = 0;
        [[nodiscard]] virtual const handle_ty *read_handle_address() const noexcept = 0;
    };

public:
    explicit Texture(Device::Impl *device, uint2 res,
                        PixelStorage pixel_storage)
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage)) {}

    /// for dsl
    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    [[nodiscard]] auto sample(const U &u, const V &v) const noexcept {
        const UniformBinding &uniform = Function::current()->get_uniform_var(Type::of<Texture<T>>(),
                                                                             read_handle(),
                                                                             Variable::Tag::TEXTURE);
        return make_expr<Texture<T>>(uniform.expression()).sample(u, v);
    }

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD auto sample(const UV &uv) const noexcept {
        return sample(uv.x, uv.y);
    }

    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] uint2 resolution() const noexcept { return impl()->resolution(); }
    [[nodiscard]] handle_ty write_handle() const noexcept { return impl()->write_handle(); }
    [[nodiscard]] handle_ty read_handle() const noexcept { return impl()->read_handle(); }
    [[nodiscard]] const handle_ty *read_handle_address() const noexcept { return impl()->read_handle_address(); }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return impl()->pixel_storage(); }
    [[nodiscard]] TextureUploadCommand *upload(const void *data) const noexcept {
        return TextureUploadCommand::create(data, write_handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return TextureUploadCommand::create(data, write_handle(), resolution(), pixel_storage(), false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data) const noexcept {
        return TextureDownloadCommand::create(data, read_handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return TextureDownloadCommand::create(data, read_handle(), resolution(), pixel_storage(), false);
    }
};

}// namespace ocarina