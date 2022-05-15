//
// Created by Zero on 12/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"
#include "expr.h"
#include "ast/function_builder.h"

namespace katana {

template<typename Lhs, typename Rhs>
void assign(Lhs &&lhs, Rhs &&rhs) noexcept;

namespace detail {

template<typename T>
struct RefEnableSubscriptAccess {
//    template<typename Index, KTN_REQUIRES(is_integral_expr_v<Index>)>
//    KTN_NODISCARD auto operator[](Index &&index) const &noexcept {
//        auto self = def<T>(static_cast<const T *>(this)->expression());
//        using Element = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
//        return def<Element>(katana::FunctionBuilder::current())
//    }
};

template<typename T>
struct RefEnableGetMemberByIndex {
};

template<typename T>
struct Ref : ExprEnableStaticCast<Ref<T>>,
             ExprEnableBitwiseCast<Ref<T>> {
    static_assert(concepts::scalar<T>);

private:
    const Expression *_expression;

public:
    explicit Ref(const Expression *e) noexcept : _expression{e} {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    Ref(Ref &&) noexcept = default;
    Ref(const Ref &) noexcept = default;
    template<typename Rhs>
    Ref &operator=(Rhs &&rhs) &noexcept {
        assign(*this, std::forward<Rhs>(rhs));
        return *this;
    }
    [[nodiscard]] explicit operator Expr<T>() const noexcept { return Expr<T>{this->expression()}; }
    Ref &operator=(Ref rhs) &noexcept {
        (*this) = Expr<T>{rhs};
        return *this;
    }

    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i == 0u);
        return *this;
    }
};

}// namespace detail

}// namespace katana