//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "resources/resource.h"
#include "command.h"

namespace ocarina {
class Accel : public RHIResource {
public:
    explicit Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, 0) {}
};
}// namespace ocarina
