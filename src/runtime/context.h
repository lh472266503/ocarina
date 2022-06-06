//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace katana {
class Context {
private:
    struct Impl;
    katana::unique_ptr<Impl> _impl;
};
}// namespace katana