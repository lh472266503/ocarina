//
// Created by Zero on 09/07/2022.
//

#include "shader.h"
#include "texture.h"

namespace ocarina {

void ArgumentList::_encode_texture(const RHITexture &texture) noexcept {
    push_memory_block(texture.memory_block());
}

void ArgumentList::_encode_resource_array(const ResourceArray &resource_array) noexcept {
    push_memory_block(resource_array.memory_block());
}

}// namespace ocarina