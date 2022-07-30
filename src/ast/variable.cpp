//
// Created by Zero on 2022/7/30.
//

#include "variable.h"
#include "symbol_name.h"

namespace ocarina {

string Variable::name() const noexcept {
    return string(detail::variable_prefix(tag())) + detail::to_string(_uid);
}
}// namespace ocarina