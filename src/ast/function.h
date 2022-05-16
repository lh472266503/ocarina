//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/stl.h"
#include "core/header.h"
#include "type.h"
#include "variable.h"

namespace katana {

namespace detail {
class FunctionBuilder;
}

class KTN_AST_API Function {
public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
    };

    struct Constant {
    };

private:
    const detail::FunctionBuilder *_builder{nullptr};

public:
    Function() noexcept = default;
    explicit Function(const detail::FunctionBuilder *builder) noexcept : _builder{builder} {}
    [[nodiscard]] katana::span<const Variable> builtin_variables() const noexcept;
    [[nodiscard]] katana::span<const Constant> constants() const noexcept;
    [[nodiscard]] katana::span<const Variable> arguments() const noexcept;
    [[nodiscard]] uint3 block_size() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
};

}// namespace katana