//
// Created by Zero on 09/07/2022.
//

#include "shader.h"
#include "texture.h"

namespace ocarina {

void ArgumentList::_encode_texture(const RHITexture &texture) noexcept {
    push_memory_block(texture.memory_block());
}
}// namespace ocarina