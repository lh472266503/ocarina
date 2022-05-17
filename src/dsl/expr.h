//
// Created by Zero on 06/05/2022.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/expr_traits.h"
#include "ast/function_builder.h"

namespace katana {

struct Expression;

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept;// implement in builtin.h

namespace detail {
template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::STATIC, src));
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto bit_cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::BITWISE, src));
    }
};

template<typename T>
struct ExprEnableSubscriptAccess {
    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] auto operator[](Index &&index) const &noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Element = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Element>(katana::FunctionBuilder::current()->access(
            Type::of<Element>(), self.expression(),
            extract_expression(std::forward<Index>(index))));
    }
};

template<typename T>
struct ExprEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        return (static_cast<const T *>(this))[static_cast<uint>(i)];
    }
};

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept {
    if constexpr (is_dsl_v<T>) {
        return std::forward<T>(v).expression();
    } else {
        return katana::FunctionBuilder::current()->literal(Type::of<T>(), std::forward<T>(v));
    }
}

}// namespace detail

#define KTN_EXPR_COMMON(...)                                               \
private:                                                                   \
    const Expression *_expression{nullptr};                                \
                                                                           \
public:                                                                    \
    explicit Expr(const Expression *e) noexcept                            \
        : _expression(e) {}                                                \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Expr(Expr &&expr) noexcept = default;                                  \
    Expr(const Expr &expr) noexcept = default;                             \
    Expr &operator=(Expr) noexcept = delete;

#define KTN_EXPR_LITERAL(...)                              \
    explicit Expr(__VA_ARGS__ &&literal) noexcept          \
        : _expression(FunctionBuilder::current()->literal( \
              Type::of<T>(), literal)) {}

/// expr for scalar
template<typename T>
struct Expr
    : detail::ExprEnableBitwiseCast<Expr<T>>,
      detail::ExprEnableStaticCast<Expr<T>> {
    static_assert(concepts::scalar<T>);
    KTN_EXPR_COMMON(T)
    KTN_EXPR_LITERAL(T)
};

template<typename T, size_t N>
struct Expr<std::array<T, N>>
    : detail::ExprEnableSubscriptAccess<std::array<T, N>>,
      detail::ExprEnableGetMemberByIndex<std::array<T, N>> {
    KTN_EXPR_COMMON(std::array<T, N>)
};

}// namespace katana