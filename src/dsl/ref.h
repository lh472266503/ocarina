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
    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] auto operator[](Index &&index) const &noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Element = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Element>(katana::FunctionBuilder::current()->access(
            Type::of<Element>(), self.expression(),
            extract_expression(std::forward<Index>(index))));
    }

    //todo
    //    template<typename Index>
    //    requires is_integral_expr_v<Index>
    //    [[nodiscard]] auto operator[](Index &&index) &noexcept {
    //    }
};

template<typename T>
struct RefEnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        auto self = const_cast<T *>(static_cast<const T *>(this));
        return Ref((*self)[static_cast<uint>(i)]);
    }
};

#define KTN_REF_COMMON(...)                                                \
private:                                                                   \
    const Expression *_expression{nullptr};                                \
                                                                           \
public:                                                                    \
    explicit Ref(const Expression *e) noexcept : _expression{e} {}         \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Ref(Ref &&) noexcept = default;                                        \
    Ref(const Ref &) noexcept = default;                                   \
    template<typename Rhs>                                                 \
    void operator=(Rhs &&rhs) &noexcept {                                  \
        assign(*this, std::forward<Rhs>(rhs));                             \
    }                                                                      \
    [[nodiscard]] explicit operator Expr<__VA_ARGS__>() const noexcept {   \
        return Expr<__VA_ARGS__>{this->expression()};                      \
    }                                                                      \
    void operator=(Ref rhs) &noexcept {                                    \
        (*this) = Expr<__VA_ARGS__>{rhs};                                  \
    }

/**
 * for scalar
 * @tparam T
 */
template<typename T>
struct Ref : ExprEnableStaticCast<Ref<T>>,
             ExprEnableBitwiseCast<Ref<T>> {
    static_assert(concepts::scalar<T>);
    KTN_REF_COMMON(T)
};

template<typename T>
struct Ref<Vector<T, 2>>
    : ExprEnableStaticCast<Ref<Vector<T, 2>>>,
      ExprEnableBitwiseCast<Ref<Vector<T, 2>>>,
      RefEnableGetMemberByIndex<Ref<Vector<T, 2>>>,
      RefEnableSubscriptAccess<Ref<Vector<T, 2>>> {
    KTN_REF_COMMON(Vector<T, 2>)
};

template<typename T>
struct Ref<Vector<T, 3>>
    : ExprEnableStaticCast<Ref<Vector<T, 3>>>,
      ExprEnableBitwiseCast<Ref<Vector<T, 3>>>,
      RefEnableGetMemberByIndex<Ref<Vector<T, 3>>>,
      RefEnableSubscriptAccess<Ref<Vector<T, 3>>> {
    KTN_REF_COMMON(Vector<T, 3>)
};

template<typename T>
struct Ref<Vector<T, 4>>
    : ExprEnableStaticCast<Ref<Vector<T, 4>>>,
      ExprEnableBitwiseCast<Ref<Vector<T, 4>>>,
      RefEnableGetMemberByIndex<Ref<Vector<T, 4>>>,
      RefEnableSubscriptAccess<Ref<Vector<T, 4>>> {
    KTN_REF_COMMON(Vector<T, 4>)
};

template<size_t N>
struct Ref<Matrix<N>>
    : ExprEnableStaticCast<Ref<Matrix<N>>>,
      ExprEnableBitwiseCast<Ref<Matrix<N>>>,
      RefEnableGetMemberByIndex<Ref<Matrix<N>>>,
      RefEnableSubscriptAccess<Ref<Matrix<N>>> {
    KTN_REF_COMMON(Matrix<N>)
};

#undef KTN_REF_COMMON

}// namespace detail

}// namespace katana