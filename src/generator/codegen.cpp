//
// Created by Zero on 06/06/2022.
//

#include "codegen.h"

namespace ocarina {

Codegen::Scratch &Codegen::Scratch::operator<<(int v) noexcept {
    return *this << detail::to_string(v);
}
Codegen::Scratch &Codegen::Scratch::operator<<(float v) noexcept {
    auto s = detail::to_string(v);
    *this << s;
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) {
        *this << ".f";
    } else if (s.find('.') != std::string::npos || s.find('e') != std::string::npos) {
        *this << "f";
    }
    return *this;
}
Codegen::Scratch &Codegen::Scratch::operator<<(bool v) noexcept {
    return *this << detail::to_string(v);
}
Codegen::Scratch &Codegen::Scratch::operator<<(ocarina::string_view v) noexcept {
    _buffer.append(v);
    return *this;
}
Codegen::Scratch &Codegen::Scratch::operator<<(const ocarina::string &v) noexcept {
    return *this << string_view{v};
}
Codegen::Scratch &Codegen::Scratch::operator<<(const char *v) noexcept {
    return *this << string_view{v};
}
Codegen::Scratch &Codegen::Scratch::operator<<(const Codegen::Scratch &scratch) noexcept {
    return *this << scratch.c_str();
}

void Codegen::Scratch::clear() noexcept {
    _buffer.clear();
}
bool Codegen::Scratch::empty() const noexcept {
    return _buffer.empty();
}
const char *Codegen::Scratch::c_str() const noexcept {
    return _buffer.c_str();
}
size_t Codegen::Scratch::size() const noexcept {
    return _buffer.size();
}
ocarina::string_view Codegen::Scratch::view() const noexcept {
    return _buffer;
}
Codegen::Scratch &Codegen::Scratch::operator<<(uint v) noexcept {
    return *this << detail::to_string(v) + "u";
}
Codegen::Scratch &Codegen::Scratch::operator<<(size_t v) noexcept {
    return *this << detail::to_string(v) + "ul";
}
void Codegen::Scratch::pop_back() noexcept {
    _buffer.pop_back();
}

void Codegen::Scratch::replace(string_view substr, string_view new_str) noexcept {
    auto begin = _buffer.find(substr);
    auto size = substr.size();
    _buffer.replace(begin, size, new_str);
}

void Codegen::_emit_newline() noexcept {
    current_scratch() << "\n";
}
void Codegen::_emit_indent() noexcept {
    static constexpr auto indent_str = "    ";
    for (int i = 0; i < _indent; ++i) {
        current_scratch() << indent_str;
    }
}
void Codegen::_emit_space() noexcept {
    current_scratch() << " ";
}

void Codegen::_emit_func_name(const Function &f) noexcept {
    current_scratch() << f.func_name();
}

void Codegen::_emit_struct_name(const Type *type) noexcept {
    current_scratch() << detail::struct_name(type->hash());
}

void Codegen::_emit_member_name(int index) noexcept {
    current_scratch() << detail::member_name(index);
}

}// namespace ocarina