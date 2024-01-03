//
// Created by Zero on 04/05/2022.
//

#pragma once

#include <cstdint>

namespace ocarina {

enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 1 << 0,
    WRITE = 1 << 1,
    READ_WRITE = READ | WRITE
};

}