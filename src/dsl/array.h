//
// Created by Zero on 17/07/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/type_registry.h"
#include "core/string_util.h"

namespace ocarina {

template<typename T>
class Array : public detail::EnableSubscriptAccess<Array<T>, T> {
public:
    using element_type = T;

private:
    size_t _size{};
    const RefExpr *_expression{};

public:
    explicit Array(size_t num)
        : _size(num) {
        const Type *type = Type::from(ocarina::format("array<{},{}>",
                                                      detail::TypeDesc<T>::description(),
                                                      num));
        _expression = Function::current()->local(type);
    }

    Array(Array &&) noexcept = default;

    Array(const Array &other) noexcept
        : _size{other._size} {
        const Type *type = Type::from(ocarina::format("array<{},{}>",
                                                      detail::TypeDesc<T>::description(),
                                                      _size));
        _expression = Function::current()->local(type);
        Function::current()->assign(_expression, other._expression);
    }

    template<typename U>
    requires is_array_expr_v<U>
    static Array<T> create(U &&array) noexcept {
        Array<T> ret{array.size()};
        for (uint i = 0; i < ret.size(); ++i) {
            ret[i] = array[i];
        }
        return ret;
    }

    Array &operator=(const Array &rhs) noexcept {
        if (&rhs != this) [[likely]] {
            OC_ASSERT(_size == rhs._size);
            Function::current()->assign(_expression, rhs._expression);
        }
        return *this;
    }
    Array &operator=(Array &&rhs) noexcept {
        *this = static_cast<const Array &>(rhs);
        return *this;
    }

    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
};

}// namespace ocarina