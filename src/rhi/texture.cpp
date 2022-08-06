//
// Created by Zero on 06/06/2022.
//

#include "texture.h"

namespace ocarina {

RHITexture::RHITexture(Device::Impl *device, uint2 res,
                       PixelStorage pixel_storage)
    : RHIResource(device, Tag::TEXTURE,
                  device->create_texture(res, pixel_storage)) {
}

TextureUploadCommand *RHITexture::upload(const void *data) const noexcept {
    return TextureUploadCommand::create(data, handle(), resolution(), pixel_storage(), true);
}

TextureUploadCommand *RHITexture::upload_sync(const void *data) const noexcept {
    return TextureUploadCommand::create(data, handle(), resolution(), pixel_storage(), false);
}

TextureDownloadCommand *RHITexture::download(void *data) const noexcept {
    return TextureDownloadCommand::create(data, handle(), resolution(), pixel_storage(), true);
}

TextureDownloadCommand *RHITexture::download_sync(void *data) const noexcept {
    return TextureDownloadCommand::create(data, handle(), resolution(), pixel_storage(), false);
}

}// namespace ocarina