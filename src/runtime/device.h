//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace nano {
class Device {
public:
    struct Impl {

    };
    using Handle = nano::unique_ptr<Impl>;

protected:
    Handle _impl;

public:
    Device();
};
}// namespace nano