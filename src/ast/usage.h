//
// Created by Zero on 04/05/2022.
//

#pragma once

#include <cstdint>

namespace sycamore {

enum struct Usage : uint32_t {
    NONE = 0u,
    READ = 0x01u,
    WRITE = 0x02u,
    READ_WRITE = READ | WRITE
};

}