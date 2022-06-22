//
// Created by Zero on 06/06/2022.
//

#include "codegen.h"

namespace ocarina {

namespace detail {

template<typename T>
[[nodiscard]] ocarina::string to_string(T &&t) noexcept {
    if constexpr (std::is_same_v<bool, std::remove_cvref_t<T>>) {
        return t ? "true" : "false";
    }
    return ocarina::to_string(std::forward<T>(t));
}

}

Codegen::Scratch &Codegen::Scratch::operator<<(int v) noexcept {
    return *this << detail::to_string(v);
}
Codegen::Scratch &Codegen::Scratch::operator<<(float v) noexcept {
    return *this << detail::to_string(v);
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
    return *this << detail::to_string(v);
}
Codegen::Scratch &Codegen::Scratch::operator<<(size_t v) noexcept {
    return *this << detail::to_string(v);
}
void Codegen::Scratch::pop_back() noexcept {
    _buffer.pop_back();
}

void Codegen::_emit_newline() noexcept {
    _scratch << "\n";
}
void Codegen::_emit_indent() noexcept {
    static constexpr auto indent_str = "    ";
    for (int i = 0; i < _indent; ++i) {
        _scratch << indent_str;
    }
}
void Codegen::_emit_space() noexcept {
    _scratch << " ";
}
}// namespace ocarina