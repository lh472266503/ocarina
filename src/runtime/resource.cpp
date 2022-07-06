//
// Created by Zero on 06/07/2022.
//

#include "resource.h"

namespace ocarina {

Resource::~Resource() {
    switch (_tag) {
        case Tag::BUFFER: _device->destroy_buffer(_handle); break;
        case Tag::TEXTURE: _device->destroy_texture(_handle); break;
    }
}
}// namespace ocarina