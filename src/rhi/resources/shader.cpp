//
// Created by Zero on 09/07/2022.
//

#include "shader.h"
#include "image.h"

namespace ocarina {

void ArgumentList::_encode_texture(const Image &texture) noexcept {
    push_handle_ptr(const_cast<handle_ty *>(texture.handle_ptr()));
}
}// namespace ocarina