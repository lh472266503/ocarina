//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "device.h"
#include "resource.h"

namespace ocarina {

template<typename... Args>
class Shader final : public Resource {
public:
    using signature = typename detail::canonical_signature_t<void(Args...)>;

private:
    ShaderTag _shader_tag;

public:
    Shader(Device::Impl *device, const Function &function, ShaderTag tag) noexcept
        : Resource(device, SHADER,
                   device->create_shader(function)),
          _shader_tag(tag) {}

    Shader &operator()(Args &&...args) noexcept {
        return *this;
    }
};

}// namespace ocarina