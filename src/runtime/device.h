//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace katana {
class Device {
public:
    struct Impl;
    using Handle = katana::unique_ptr<Impl>;

private:
    Handle _impl;

public:
    Device() {

    }
};
}// namespace katana