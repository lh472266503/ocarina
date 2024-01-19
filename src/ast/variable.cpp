//
// Created by Zero on 2022/7/30.
//

#include "variable.h"
#include "symbol_name.h"

namespace ocarina {

string Variable::name() const noexcept {
    string raw_name = string(detail::variable_prefix(tag())) + detail::to_string(_uid);
    if (_name) { return _name; }
    if (_suffix) {
        return raw_name + "_" + _suffix;
    }
    return raw_name;
}
}// namespace ocarina