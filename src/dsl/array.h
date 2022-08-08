//
// Created by Zero on 17/07/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"

namespace ocarina {

template<typename T>
class Array : public detail::EnableSubscriptAccess<std::array<T, 1>> {
public:
    using element_type = T;

private:
    const RefExpr *_expression{};
    size_t _size{};

public:
    explicit Array(size_t num) {}
};

}// namespace ocarina