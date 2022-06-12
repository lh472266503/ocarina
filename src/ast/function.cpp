//
// Created by Zero on 30/04/2022.
//

#include "function.h"
#include "function_builder.h"

namespace ocarina {

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

const Type *Function::return_type() const noexcept {
    return _builder->return_type();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

bool Function::is_callable() const noexcept {
    return _builder->is_callable();
}

bool Function::is_kernel() const noexcept {
    return _builder->is_kernel();
}
}// namespace ocarina