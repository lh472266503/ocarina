//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "resource.h"
#include "command.h"

namespace ocarina {
class Accel : public RHIResource {
public:
    Accel(Device::Impl *device)
        : RHIResource(device, Tag::ACCEL, 0) {}
};
}// namespace ocarina
