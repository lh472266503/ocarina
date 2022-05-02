//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"

namespace sycamore {
namespace ast {

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
private:
    const Type *_type;
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

private:
    SCM_NODISCARD virtual uint64_t _compute_hash() const noexcept = 0;

public:
    SCM_NODISCARD uint64_t hash() const noexcept;
};

}
}// namespace sycamore::ast