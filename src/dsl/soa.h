//
// Created by Zero on 2024/2/25.
//

#pragma once

#include "core/basic_types.h"
#include "var.h"

namespace ocarina {

template<typename T>
struct SOAView {
    static_assert(always_false_v<T>);
};

template<>
struct SOAView<uint> {
    ByteBufferVar buffer;
};

}
