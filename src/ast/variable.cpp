//
// Created by Zero on 2022/7/30.
//

#include "variable.h"
#include "symbol_name.h"

namespace ocarina {

uint64_t Variable::_compute_hash() const noexcept {
    auto u0 = static_cast<uint64_t>(_uid);
    auto u1 = static_cast<uint64_t>(_tag);
    uint64_t ret = hash64(u0 | (u1 << 32u), type()->hash());
    if (_name) {
        ret = hash64(ret, _name);
    }
    if (_suffix) {
        ret = hash64(ret, _suffix);
    }
    return ret;
}

string Variable::name() const noexcept {
    string raw_name = string(detail::variable_prefix(tag())) + detail::to_string(_uid);
    if (_name) { return _name; }
    if (_suffix) {
        return raw_name + "_" + _suffix;
    }
    return raw_name;
}
}// namespace ocarina