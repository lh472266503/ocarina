//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "core/logging.h"

namespace ocarina {

namespace detail {

template<typename T>
[[nodiscard]] ocarina::string to_string(T &&t) noexcept {
    if constexpr (std::is_same_v<bool, std::remove_cvref_t<T>>) {
        return t ? "true" : "false";
    } else if constexpr (std::is_same_v<float, std::remove_cvref_t<T>>) {
        return ocarina::to_string(std::forward<T>(t)) + "f";
    }
    return ocarina::to_string(std::forward<T>(t));
}

struct LiteralPrinter;

[[nodiscard]] inline string struct_name(uint64_t hash) {
    return "structure_" + to_string(hash);
}

[[nodiscard]] inline string func_name(uint64_t hash) {
    return "function_" + to_string(hash);
}

}// namespace detail

class Codegen {
protected:
    class Scratch {
    private:
        ocarina::string _buffer;

    public:
        Scratch() = default;
        explicit Scratch(const string_view &str) noexcept { _buffer = str; }
        Scratch &operator<<(ocarina::string_view v) noexcept;
        Scratch &operator<<(const ocarina::string &v) noexcept;
        Scratch &operator<<(const char *v) noexcept;
        Scratch &operator<<(int v) noexcept;
        Scratch &operator<<(float v) noexcept;
        Scratch &operator<<(bool v) noexcept;
        Scratch &operator<<(uint v) noexcept;
        Scratch &operator<<(size_t v) noexcept;
        Scratch &operator<<(const Scratch &scratch) noexcept;
        void replace(int index, string_view substr, string_view new_str) noexcept;
        void clear() noexcept;
        void pop_back() noexcept;
        [[nodiscard]] const char *c_str() const noexcept;
        [[nodiscard]] ocarina::string_view view() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
    };

private:
    int _indent{};
    Scratch _scratch;
    ocarina::vector<Scratch *> _scratch_stack;

protected:
    void indent_inc() noexcept { _indent += 1; }
    void indent_dec() noexcept { _indent -= 1; }
    void push_scratch(Scratch &scratch) noexcept { _scratch_stack.push_back(&scratch); }
    void pop_scratch(Scratch &scratch) noexcept {
        Scratch *back = _scratch_stack.back();
        if (&scratch != back) [[unlikely]] {
            OC_ERROR("Invalid scratch !");
        }
        _scratch_stack.pop_back();
    }
    Scratch &current_scratch() noexcept { return *_scratch_stack.back(); }
    friend struct detail::LiteralPrinter;

    class ScratchGuard {
    private:
        Scratch &_scratch;
        Codegen *_codegen{};
    public:
        ScratchGuard(Codegen *codegen, Scratch &scratch)
            : _codegen(codegen), _scratch(scratch) {
            _codegen->push_scratch(_scratch);
        }
        ~ScratchGuard() {
            _codegen->pop_scratch(_scratch);
        }
    };
#define SCRATCH_GUARD(scratch) ScratchGuard __##scratch_guard(this, scratch);

protected:
    virtual void _emit_newline() noexcept;
    virtual void _emit_indent() noexcept;
    virtual void _emit_space() noexcept;

    virtual void _emit_func_name(uint64_t hash) noexcept;
    virtual void _emit_struct_name(uint64_t hash) noexcept;
    virtual void _emit_member_name(int index) noexcept;

public:
    Codegen() { push_scratch(_scratch); }
    explicit Codegen(Scratch &scratch)
        : _scratch(scratch) {
        push_scratch(_scratch);
    }
    virtual void emit(const Function &func) = 0;
    Scratch &scratch() {
        return _scratch;
    }
};

}// namespace ocarina