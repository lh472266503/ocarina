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
class Array : public detail::EnableSubscriptAccess<Array<T>, T>,
              public concepts::Noncopyable {
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
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expression; }
    [[nodiscard]] size_t size() const noexcept { return _size; }
};

}// namespace ocarina