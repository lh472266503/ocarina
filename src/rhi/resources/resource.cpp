//
// Created by Zero on 06/07/2022.
//

#include "resource.h"

namespace ocarina {

void RHIResource::_destroy() {
    if (!valid()) { return; }
    switch (_tag) {
        case Tag::BUFFER: _device->destroy_buffer(_handle); break;
        case Tag::TEXTURE: _device->destroy_texture(_handle); break;
        case Tag::STREAM: _device->destroy_stream(_handle); break;
        case Tag::SHADER: _device->destroy_shader(_handle); break;
        case Tag::MESH: _device->destroy_mesh(_handle); break;
        case Tag::ACCEL: _device->destroy_accel(_handle); break;
    }
}
}// namespace ocarina