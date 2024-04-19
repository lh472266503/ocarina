//
// Created by Zero on 2022/7/30.
//

#include "variable.h"
#include "symbol_name.h"
#include "function.h"

namespace ocarina {

Variable::Variable(const Function *context,
                   const Type *type,
                   Variable::Tag tag,
                   uint uid, std::string name,
                   std::string suffix) noexcept
    : _context(context), type_(type), uid_(uid) {
    set_tag(tag);
    set_suffix(std::move(suffix));
    set_name(std::move(name));
}

uint64_t Variable::_compute_hash() const noexcept {
    auto u0 = static_cast<uint64_t>(uid_);
    auto u1 = static_cast<uint64_t>(tag());
    uint64_t ret = hash64(u0 | (u1 << 32u), type()->hash());
    if (!get_name().empty()) {
        ret = hash64(ret, get_name());
    }
    if (!get_suffix().empty()) {
        ret = hash64(ret, get_suffix());
    }
    return ret;
}

string Variable::get_suffix() const noexcept {
    return _context->variable_data(uid_).suffix;
}

string Variable::get_name() const noexcept {
    return _context->variable_data(uid_).name;
}

void Variable::set_name(string name) noexcept {
    const_cast<Function *>(_context)->variable_data(uid_).name = std::move(name);
}

void Variable::set_suffix(std::string suffix) noexcept {
    const_cast<Function *>(_context)->variable_data(uid_).suffix = std::move(suffix);
}

Variable::Tag Variable::tag() const noexcept {
    return _context->variable_data(uid_).tag;
}

void Variable::set_tag(ocarina::Variable::Tag tag) noexcept {
    const_cast<Function *>(_context)->variable_data(uid_).tag = tag;
}

Usage Variable::usage() const noexcept {
    return _context->variable_data(uid_).usage;
}

string Variable::name() const noexcept {
    string raw_name = string(detail::variable_prefix(tag())) + detail::to_string(uid_);
    if (!get_name().empty()) { return get_name(); }
    if (!get_suffix().empty()) {
        return raw_name + "_" + get_suffix();
    }
    return raw_name;
}
}// namespace ocarina