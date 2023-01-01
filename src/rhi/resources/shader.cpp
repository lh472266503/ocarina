//
// Created by Zero on 09/07/2022.
//

#include "shader.h"
#include "texture.h"

namespace ocarina {

void ArgumentList::_encode_image(const RHITexture &image) noexcept {
    push_memory_block(image.memory_block());
}
}// namespace ocarina