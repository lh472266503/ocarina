//
// Created by Zero on 09/07/2022.
//

#pragma once

#include "device.h"
#include "resource.h"

namespace ocarina {

template<size_t dimension, typename... Args>
class Shader final : public Resource {
public:
    enum Tag {
        CS,
        VS,
        FS,
        GS,
        TS
    };

private:
    Tag _tag;

public:
    Shader(Device::Impl *device, handle_ty handle, Tag tag = CS) noexcept
        : Resource(device, SHADER, handle), _tag(tag) {}

    Shader &operator()(Args &&...args) noexcept {
        return *this;
    }
};

template<typename... Args>
using Shader1D = Shader<1, Args...>;

template<typename... Args>
using Shader2D = Shader<2, Args...>;

template<typename... Args>
using Shader3D = Shader<3, Args...>;

}// namespace ocarina