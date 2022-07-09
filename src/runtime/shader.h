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
    explicit Shader(Device::Impl *device) {

    }
};

}// namespace ocarina