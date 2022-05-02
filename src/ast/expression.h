//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"

namespace sycamore {
inline namespace ast {

class Expression : public concepts::Noncopyable {
public:
    enum struct Tag : uint32_t {
        UNARY,
        BINARY,
        MEMBER,
        ACCESS,
        LITERAL,
        REF,
        CONSTANT,
        CALL,
        CAST
    };

};

}
}// namespace sycamore::ast