//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/header.h"
#include "core/concepts.h"

namespace sycamore {
inline namespace ast {

class SCM_AST_API Statement : public concepts::Noncopyable {
public:
    /// Statement types
    enum struct Tag : uint32_t {
        BREAK,
        CONTINUE,
        RETURN,
        SCOPE,
        IF,
        LOOP,
        EXPR,
        SWITCH,
        SWITCH_CASE,
        SWITCH_DEFAULT,
        ASSIGN,
        FOR,
        COMMENT,
        META
    };
};

}
}// namespace sycamore::ast