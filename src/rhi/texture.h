//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"
#include "command.h"

namespace ocarina {

template<typename T>
class RHITexture : public RHIResource {
public:
    class Impl {
    public:
        [[nodiscard]] virtual uint2 resolution() const noexcept = 0;
        [[nodiscard]] virtual PixelStorage pixel_storage() const noexcept = 0;
        [[nodiscard]] virtual handle_ty handle() const noexcept = 0;
    };

public:
    explicit RHITexture(Device::Impl *device, uint2 res,
                        PixelStorage pixel_storage)
        : RHIResource(device, Tag::TEXTURE,
                      device->create_texture(res, pixel_storage)) {}
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] uint2 resolution() const noexcept { return impl()->resolution(); }
    [[nodiscard]] handle_ty handle() const noexcept { return impl()->handle(); }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return impl()->pixel_storage(); }
    [[nodiscard]] TextureUploadCommand *upload(const void *data) const noexcept {
        return TextureUploadCommand::create(data, handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureUploadCommand *upload_sync(const void *data) const noexcept {
        return TextureUploadCommand::create(data, handle(), resolution(), pixel_storage(), false);
    }
    [[nodiscard]] TextureDownloadCommand *download(void *data) const noexcept {
        return TextureDownloadCommand::create(data, handle(), resolution(), pixel_storage(), true);
    }
    [[nodiscard]] TextureDownloadCommand *download_sync(void *data) const noexcept {
        return TextureDownloadCommand::create(data, handle(), resolution(), pixel_storage(), false);
    }
};

}// namespace ocarina