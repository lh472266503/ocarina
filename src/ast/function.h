//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/stl.h"
#include "core/header.h"
#include "type.h"
#include "variable.h"

namespace ocarina {

class FunctionBuilder;

class OC_AST_API Function {
public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
    };

private:
    const FunctionBuilder *_builder{nullptr};

private:
    class Impl;
    ocarina::shared_ptr<Impl> _impl{};

private:
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto builder = ocarina::make_shared<FunctionBuilder>(tag);
        auto ret = Function(tag);
        func();
        return ret;
    }

public:
    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }

    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }
    Function() noexcept = default;
    explicit Function(const FunctionBuilder *builder) noexcept : _builder{builder} {}
    explicit Function(Tag tag) noexcept;
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] bool is_callable() const noexcept;
    [[nodiscard]] bool is_kernel() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
};

}// namespace ocarina