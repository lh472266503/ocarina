//
// Created by Zero on 2024/3/22.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include <glad/glad.h>

namespace ocarina {

class Texture;

class GLInterop {
protected:
    GLuint _pbo{0u};
    Texture *_texture{};

public:
    explicit GLInterop(Texture *texture)
        : _texture(texture) {}

    virtual void register_() noexcept = 0;
    virtual void mapping() noexcept = 0;
    virtual void unmapping() noexcept = 0;
    virtual void unregister_() noexcept = 0;
    virtual GLuint get_pbo() noexcept = 0;
};

}// namespace ocarina