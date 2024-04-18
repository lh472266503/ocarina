//
// Created by Zero on 2022/7/30.
//

#include "variable.h"
#include "symbol_name.h"

namespace ocarina {

uint64_t Variable::_compute_hash() const noexcept {
    auto u0 = static_cast<uint64_t>(uid_);
    auto u1 = static_cast<uint64_t>(tag_);
    uint64_t ret = hash64(u0 | (u1 << 32u), type()->hash());
    if (!name_.empty()) {
        ret = hash64(ret, name_);
    }
    if (!suffix_.empty()) {
        ret = hash64(ret, suffix_);
    }
    return ret;
}

string Variable::name() const noexcept {
    string raw_name = string(detail::variable_prefix(tag())) + detail::to_string(uid_);
    if (!name_.empty()) { return name_; }
    if (!suffix_.empty()) {
        return raw_name + "_" + suffix_;
    }
    return raw_name;
}
}// namespace ocarina