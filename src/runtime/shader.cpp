//
// Created by Zero on 09/07/2022.
//

#include "shader.h"

namespace ocarina {

ShaderDispatchCommand *ShaderInvoke::dispatch(uint x, uint y, uint z) {
    return ShaderDispatchCommand::create(std::move(_args), uint3(x, y, z));
}
}