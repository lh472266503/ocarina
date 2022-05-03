//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"

namespace sycamore::ast {

class Variable {
public:
    enum struct Tag : uint32_t {
        // data
        LOCAL,
        SHARED,

        // reference
        REFERENCE,

        // resources
        BUFFER,
        TEXTURE,
        BINDLESS_ARRAY,
        ACCEL,

        // builtins
        THREAD_ID,
        BLOCK_ID,
        DISPATCH_ID,
        DISPATCH_SIZE
    };

private:
    const Type *_type;
    uint32_t _uid;
    Tag _tag;
};

}// namespace sycamore::ast