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
    return nullptr;
}
TextureDownloadCommand *RHITexture::download(const void *data) const noexcept {
    return nullptr;
}
}