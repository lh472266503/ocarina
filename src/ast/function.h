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

public:
    Function() noexcept = default;
    explicit Function(const FunctionBuilder *builder) noexcept : _builder{builder} {}
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] bool is_callable() const noexcept;
    [[nodiscard]] bool is_kernel() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;

};

}// namespace ocarina