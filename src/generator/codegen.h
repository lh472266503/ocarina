//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"

namespace ocarina {

namespace detail {
struct LiteralPrinter;
}

class Codegen {
private:
    class Scratch {
    private:
        ocarina::string _buffer;

    public:
        Scratch() = default;
        Scratch &operator<<(ocarina::string_view v) noexcept;
        Scratch &operator<<(const ocarina::string &v) noexcept;
        Scratch &operator<<(const char *v) noexcept;
        Scratch &operator<<(int v) noexcept;
        Scratch &operator<<(float v) noexcept;
        Scratch &operator<<(bool v) noexcept;
        Scratch &operator<<(uint v) noexcept;
        Scratch &operator<<(size_t v) noexcept;
        void clear() noexcept;
        void pop_back() noexcept;
        [[nodiscard]] const char *c_str() const noexcept;
        [[nodiscard]] ocarina::string_view view() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
    };

protected:
    Scratch _scratch;
    int _indent{};
    friend struct detail::LiteralPrinter;

protected:
    virtual void _emit_newline() noexcept;
    virtual void _emit_indent() noexcept;
    virtual void _emit_space() noexcept;

    virtual void _emit_func_name(uint64_t hash) noexcept;
    virtual void _emit_struct_name(uint64_t hash) noexcept;
public:
    Codegen() = default;
    explicit Codegen(Scratch &scratch)
        : _scratch(scratch) {}
    virtual void emit(const Function &func) = 0;
    Scratch &scratch() {
        return _scratch;
    }
};

}// namespace ocarina